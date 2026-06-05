# === 핵심 하이퍼파라미터 ===
NUM_CLASSES, IVD_START = 9, 3
IN_CHANNELS = 3              # T1 + T2 + (T1+T2)/2
IMG_SIZE, BATCH_SIZE, NUM_EPOCHS = 512, 8, 100
LR_ENCODER, LR_DECODER = 1e-4, 1e-3      # 차등 학습률
WEIGHT_DECAY, PATIENCE, DICE_WEIGHT = 1e-4, 25, 0.5
ENCODER_NAME = 'resnet34'

# === 증강 (T1·T2에 동일한 기하 변환 적용) ===
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, border_mode=0, p=0.5),
    A.ElasticTransform(alpha=80, sigma=10, border_mode=0, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.2, border_mode=0, p=0.2),
    A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
], additional_targets={'image_t1': 'image'})

class SpineSegDataset_T1T2(Dataset):
    """T1·T2 슬라이스를 로드해 [T1, T2, (T1+T2)/2] 3채널로 구성"""
    def __getitem__(self, idx):
        t1, t2, mask = self.images_t1[idx], self.images_t2[idx], self.masks[idx]
        if self.transform: ...   # T1·T2 동일 변환
        avg = (t1 + t2) / 2.0
        return torch.FloatTensor(np.stack([t1, t2, avg], 0)), torch.LongTensor(mask)

# IVD 포함 슬라이스에 가중치를 주는 WeightedRandomSampler (0개 0.3 / 1~3개 0.7 / 4~6개 1.0)

# === 모델: SMP U-Net + ResNet34 (ImageNet pretrained, SCSE 디코더) ===
model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet',
                 in_channels=3, classes=9, decoder_attention_type='scse').to(device)

# === Loss: Weighted CE + Dice (배경 제외, IVD 중심) ===
class DiceCELoss(nn.Module):
    def __init__(self, ce_weights, dice_weight=0.5, smooth=1e-5):
        super().__init__(); self.ce = nn.CrossEntropyLoss(weight=ce_weights)
        self.dice_weight, self.smooth = dice_weight, smooth
    def dice_loss(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()
        dice = 0.0
        for c in range(1, pred.shape[1]):       # 클래스 1~8만 (배경 제외)
            p, t = pred_soft[:,c].flatten(), target_oh[:,c].flatten()
            dice += (2*(p*t).sum()+self.smooth) / (p.sum()+t.sum()+self.smooth)
        return 1.0 - dice / (pred.shape[1]-1)
    def forward(self, pred, target):
        return (1-self.dice_weight)*self.ce(pred,target) + self.dice_weight*self.dice_loss(pred,target)

# CE 가중치: 클래스 빈도 역수(inverse frequency)로 산출
# === Optimizer / Scheduler ===
optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(),           'lr': LR_ENCODER},
    {'params': model.decoder.parameters(),           'lr': LR_DECODER},
    {'params': model.segmentation_head.parameters(), 'lr': LR_DECODER},
], weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
scaler = GradScaler()        # Mixed precision

# === 학습 루프 (Val IVD Dice 최고 시 best_unet.pth 저장, patience 25 early stop) ===
for epoch in range(1, NUM_EPOCHS+1):
    train_loss, train_dice = train_one_epoch(...)   # autocast + GradScaler
    val_loss, val_dice = validate(...)
    scheduler.step()
    val_dice_ivd = val_dice[IVD_START:].mean()       # IVD(3~8) 평균 Dice 기준
    if val_dice_ivd > best_val_dice:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'val_dice_ivd': val_dice_ivd, 'encoder_name': ENCODER_NAME,
                    'input_channels': 'T1+T2+avg'}, MODEL_DIR/'best_unet.pth')
    # early stopping ...

# === 테스트 평가: 클래스별 Dice·IoU 산출 → test_results.json 저장 ===
# 결과: Test IVD Mean Dice ≈ 0.837 / IoU ≈ 0.738 (Best epoch 74)