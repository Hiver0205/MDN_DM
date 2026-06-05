# === 핵심 하이퍼파라미터 ===
EPOCHS, PATIENCE, BATCH_SIZE = 80, 20, 16
LR, WD = 1e-4, 1e-2
FOCAL_ALPHA, FOCAL_GAMMA = 0.6, 2.0    # 양성 가중 / hard example mining
SOFT_WEIGHT = 0.3                      # Soft MaskedGAP 배경 가중치
LAMBDA_ATTN, MIXUP_ALPHA, GRAD_CLIP = 0.5, 0.3, 1.0
# 입력: (B, S, 2[T1·T2], 128, 128) 가변 슬라이스 → collate_fn으로 max_slices 패딩

# === Soft MaskedGAP: 디스크 1.0 / 배경 0.3 가중 평균 풀링 ===
class SoftMaskedGAP(nn.Module):
    def __init__(self, feat_map_size=8, soft_weight=0.3):
        super().__init__(); self.fs, self.sw = feat_map_size, soft_weight
    def forward(self, feat, disc_mask):
        ms = F.adaptive_avg_pool2d(disc_mask, self.fs)
        w = ms * (1 - self.sw) + self.sw          # 디스크 1.0, 배경 0.3 (context 유지)
        return (feat * w).sum(dim=(2,3)) / (w.sum(dim=(2,3)) + 1e-8)

# === 슬라이스 중요도 가중합 ===
class SliceAttentionAggregator(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(feat_dim,64), nn.Tanh(), nn.Linear(64,1))
    def forward(self, x, valid_mask):
        scores = self.attn(x).squeeze(-1).masked_fill(~valid_mask, -1e9)
        weights = F.softmax(scores, dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1), weights

# === 모델: ResNet18(layer3까지) + Attention Branch + SoftMaskedGAP + SliceAttention ===
class Spine25DCNN(nn.Module):
    def __init__(self, num_classes=1, soft_weight=0.3):
        super().__init__()
        pretrained = models.resnet18(weights='IMAGENET1K_V1')
        # conv1을 2채널로 교체(기존 3ch 가중치 평균→2회 복제), conv1·bn1·layer1 freeze
        self.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
        self.conv1.weight.data = pretrained.conv1.weight.data.mean(1, keepdim=True).repeat(1,2,1,1)
        self.layer1, self.layer2, self.layer3 = pretrained.layer1, pretrained.layer2, pretrained.layer3
        self.attn_branch = nn.Sequential(nn.Conv2d(128,64,1), nn.BatchNorm2d(64), nn.ReLU(),
                                         nn.Conv2d(64,1,1), nn.Sigmoid())   # 16×16 gating
        self.soft_masked_gap = SoftMaskedGAP(8, soft_weight)
        self.slice_agg = SliceAttentionAggregator(256)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, num_classes))
    def forward(self, x, mask_8x8, valid_mask):
        # 슬라이스별 feat 추출 → attention gating → SoftMaskedGAP → 슬라이스 가중합 → 분류
        ...
        return logits, attn_maps, slice_weights

# === Loss: Focal + Attention Dice(λ=0.5) ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0): super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits); pt = targets*p + (1-targets)*(1-p)
        at = targets*self.alpha + (1-targets)*(1-self.alpha)
        return (at * (1-pt)**self.gamma * bce).mean()

# === 차등 LR(layer2 ×0.1, layer3 ×0.5, attn·head ×1.0), AdamW + CosineAnnealing ===
# === 학습 기법: Mixup(α=0.3), WeightedRandomSampler(클래스 균형), Elastic/Noise/Flip 증강 ===
# === 추론: H-flip TTA, Youden's J 임계값 / 검증: 5-Fold + OOF (별도 스크립트) ===
# 출력: experiments/bulging/best_model.pt, results.json