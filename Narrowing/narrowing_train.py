# === 핵심 설정 (Config) ===
IN_CHANNELS, BASE_CH, FEAT_MAP_SIZE = 2, 16, 4
DROPOUT, SOFT_WEIGHT = 0.3, 0.3
EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY = 100, 8, 1e-3, 1e-4
PATIENCE, GRAD_CLIP = 20, 1.0
FOCAL_ALPHA, FOCAL_GAMMA = 0.6, 2.0
# 입력: (B, N, 2[T1·T2], 128, 128) 가변 슬라이스 → collate_fn 패딩
# 디스크 마스크: mask>150을 디스크 영역으로 이진화

# === 경량 백본: 5-stage ConvBlock(Conv3x3-BN-ReLU×2) + MaxPool, 16→256 채널 ===
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
    def forward(self, x): return self.block(x)

class LightweightBackbone(nn.Module):
    def __init__(self, in_channels=2, base_ch=16):
        super().__init__()
        ch = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16]   # 16→32→64→128→256
        self.stage1..5 = ConvBlock(...); self.pool1..5 = nn.MaxPool2d(2)
        self.feat_dim = ch[4]   # 256, 최종 4×4

# === MaskedGAP: 디스크 1.0 / 배경 0.3 가중 평균 풀링 (4×4 해상도) ===
class MaskedGAP(nn.Module):
    def __init__(self, feat_map_size=4, soft_weight=0.3):
        super().__init__(); self.fs, self.sw = feat_map_size, soft_weight
    def forward(self, feat, dm):
        ms = F.adaptive_avg_pool2d(dm, self.fs)
        w = ms * (1 - self.sw) + self.sw
        return (feat * w).sum(dim=(2,3)) / (w.sum(dim=(2,3)) + 1e-8)

# === 슬라이스 Attention Pooling ===
class AttentionPooling(nn.Module):
    def __init__(self, d):
        super().__init__(); self.attn = nn.Sequential(nn.Linear(d, d//4), nn.Tanh(), nn.Linear(d//4, 1))
    def forward(self, f, m):
        s = self.attn(f).squeeze(-1).masked_fill(m == 0, float('-inf'))
        w = F.softmax(s, dim=1).unsqueeze(-1)
        return (f * w).sum(dim=1), w.squeeze(-1)

# === 전체 모델 ===
class DiscModel(nn.Module):
    def __init__(self, in_ch=2, base_ch=16, feat_map_size=4, dropout=0.3, soft_weight=0.3):
        super().__init__()
        self.backbone = LightweightBackbone(in_ch, base_ch)
        self.masked_gap = MaskedGAP(feat_map_size, soft_weight)
        self.attention = AttentionPooling(self.backbone.feat_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.feat_dim, 1))
    def forward(self, x, am, dm=None):
        B, N, C, H, W = x.shape
        fm = self.backbone(x.view(B*N, C, H, W))
        pooled = self.masked_gap(fm, dm.view(B*N,1,H,W)).view(B, N, -1)
        agg, aw = self.attention(pooled, am)
        return self.classifier(agg).squeeze(-1), aw

# === Loss / Optimizer ===
criterion = FocalLoss(alpha=0.6, gamma=2.0)          # 위 팽윤과 동일 정의
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# === 학습: WeightedRandomSampler(클래스 균형), Flip/Rotation/Elastic/Noise 증강,
#           Val AUC 최고 시 best_model.pth 저장, patience 20 early stop ===
# === 평가: 검증셋 Youden's J로 임계값 탐색 → 테스트 평가 → results.json ===