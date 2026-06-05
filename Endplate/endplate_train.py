# === 핵심 설정 ===
IMG_SIZE, NUM_CLASSES, N_FOLDS = 128, 2, 5      # [upper_ep, lower_ep]
BATCH_SIZE, TOP_K, SEED = 32, 5, 42             # IVD 면적 상위 5슬라이스 선별
STAGE_A_EPOCHS, SUPCON_TEMP = 0, 0.07           # SupCon pretrain (현재 OFF)
STAGE_B_EPOCHS, STAGE_B_LR = 50, 2e-5           # 분류 학습
AUC_LOSS_WEIGHT, PATIENCE = 0.3, 10
# 입력: 8채널 = T1×3 + T2×3(2.5D 인접) + IVD 마스크 + EP 마스크

# === Multi-Scale Squeeze-Excitation 블록 ===
class SEBlock(nn.Module):  # 채널별 가중치 재조정 (AdaptiveAvgPool → FC → sigmoid)
    ...

# === 모델: EfficientNet-B0 + Multi-Scale SE + IVD 임베딩 ===
class EndplateNetV9(nn.Module):
    def __init__(self, in_ch=8, num_classes=2, num_ivd=16, ivd_dim=16, proj_dim=128, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True,
                                          in_chans=in_ch, features_only=True)
        feat = self.backbone.feature_info.channels()        # [16,24,40,112,320]
        self.se_mid, self.se_high, self.se_top = SEBlock(feat[2]), SEBlock(feat[3]), SEBlock(feat[4])
        self.pool = nn.AdaptiveAvgPool2d(1)
        fused_dim = feat[2]+feat[3]+feat[4]                  # 40+112+320 = 472
        self.ivd_emb = nn.Embedding(num_ivd, ivd_dim)        # IVD 위치 임베딩
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(fused_dim+ivd_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(True), nn.Dropout(dropout*0.5), nn.Linear(256, num_classes))
        self.projector = nn.Sequential(...)                  # SupCon용 projection head
    def extract_features(self, x):
        feats = self.backbone(x)
        # 3개 스케일(16×16·8×8·4×4)에 SE 적용 후 pooling → concat
        return torch.cat([self.pool(self.se_mid(feats[2])).flatten(1),
                          self.pool(self.se_high(feats[3])).flatten(1),
                          self.pool(self.se_top(feats[4])).flatten(1)], dim=1)
    def forward(self, x, ivd_idx):
        feat = self.extract_features(x)
        cls_feat = torch.cat([feat, self.ivd_emb(ivd_idx)], dim=1)  # IVD 임베딩 결합
        return self.classifier(cls_feat)                            # (B, 2)

# === Loss 1: Asymmetric Focal (FP에 gamma_neg=4 강한 페널티) ===
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=2.0, gamma_neg=4.0): ...
    def forward(self, logits, targets, pos_weight=None):
        ce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
        p_t = torch.sigmoid(logits)*targets + (1-torch.sigmoid(logits))*(1-targets)
        gamma = self.gamma_pos*targets + self.gamma_neg*(1-targets)
        return ((1-p_t)**gamma * ce).mean()

# === Loss 2: AUC Pairwise (양성 점수가 음성보다 margin 이상 높도록) ===
class AUCPairwiseLoss(nn.Module):
    def __init__(self, margin=1.0): ...
    def forward(self, logits, targets):
        # 클래스별 (pos, neg) 쌍에 대해 clamp(margin - (pos-neg), 0) 평균
        ...

# (Loss 3: SupConLoss(temperature=0.07) — Stage A 대조학습, 현재 미사용)

# === 학습: StratifiedGroupKFold(K=5, group=환자), Stage B 손실 = 0.7·Focal + 0.3·AUC ===
#   AdamW(2e-5, WD 5e-3), CosineAnnealingWarmRestarts, CutMix(25%), patience 10
#   fold별 best 저장 → 5-Fold 앙상블 + TTA(flip+밝기 4종)
# === 평가: Youden's J로 상/하 임계값 결정, LayerCAM·Occlusion 시각화 ===
# 출력: model_fold0~4.pth, all_metrics.csv, roc_curves.png 등