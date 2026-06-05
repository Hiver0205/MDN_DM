# === 핵심 파라미터 ===
MARGIN_MULT      = 0.15    # 디스크 크롭 수평 마진
VERT_MARGIN_MULT = 0.20    # 수직 마진 (상하 척추체 context 확보)
MIN_CROP, TARGET_CROP = 40, 128
DILATION_ITERATIONS = 3    # 마스크 팽창 반복
NORM_LOWER_PCTL, NORM_UPPER_PCTL = 1, 99   # per-slice 정규화 백분위
APPLY_BG_MASK = True       # 배경 마스킹(환자 편향 감소)
MIN_MASK_RATIO, MIN_STD_NORM, MIN_SLICES_KEEP = 0.15, 0.02, 3  # 슬라이스 필터

def disc_centered_crop(patch, mask, margin_h=MARGIN_MULT, margin_v=VERT_MARGIN_MULT,
                       min_crop=MIN_CROP, target_size=TARGET_CROP):
    """디스크 중심 비대칭 마진 재크롭 → 정사각형 → 128×128 리사이즈"""
    N, C, H, W = patch.shape
    disc_region = np.zeros((H, W), dtype=bool)
    for s in range(N):
        disc_region |= (mask[s] > 0.5)        # 전체 슬라이스 디스크 영역 합집합
    ys, xs = np.where(disc_region)
    cy, cx = (ys.min()+ys.max())/2, (xs.min()+xs.max())/2
    disc_h, disc_w = ys.ptp()+1, xs.ptp()+1
    # 비대칭 마진 적용 후 정사각형 크롭 크기 결정
    crop = max(disc_h + 2*int(disc_h*margin_v), disc_w + 2*int(disc_w*margin_h), min_crop)
    crop = min(crop, H, W)
    # 중심 기준 crop → target_size로 zoom (이미지 order=1 / 마스크 order=0)
    ...
    return cropped_patch, cropped_mask, crop_info

def dilate_masks(mask, iterations=DILATION_ITERATIONS):
    """디스크 마스크 binary dilation (배경마스킹 시 경계 여유 확보)"""
    struct = generate_binary_structure(2, 1)
    dilated = np.zeros_like(mask, dtype=np.float32)
    for i in range(mask.shape[0]):
        if mask[i].max() > 0:
            dilated[i] = binary_dilation(mask[i] > 0.5, structure=struct,
                                         iterations=iterations).astype(np.float32)
    return dilated

def per_slice_normalize(patch, lower=NORM_LOWER_PCTL, upper=NORM_UPPER_PCTL):
    """슬라이스·채널별 percentile[1,99] 클립 후 [0,1] 정규화"""
    normed = np.zeros_like(patch, dtype=np.float32)
    for i in range(patch.shape[0]):
        for ch in range(patch.shape[1]):
            sl = patch[i, ch]
            p_low, p_high = np.percentile(sl, lower), np.percentile(sl, upper)
            if p_high - p_low >= 1e-6:
                normed[i, ch] = (np.clip(sl, p_low, p_high) - p_low) / (p_high - p_low)
    return normed

def apply_background_masking(patch, mask, soft_edge=True):
    """dilated 마스크 외부를 0으로 (soft-edge gaussian σ=2.0) → 환자 배경 편향 제거"""
    masked = patch.copy()
    for i in range(patch.shape[0]):
        soft = gaussian_filter(mask[i].astype(np.float32), sigma=2.0)
        soft = np.clip(soft / max(soft.max(), 1e-8), 0, 1)
        for ch in range(patch.shape[1]):
            masked[i, ch] *= soft
    return masked

def filter_slices(patch, mask, min_mask_ratio=MIN_MASK_RATIO,
                  min_std=MIN_STD_NORM, min_keep=MIN_SLICES_KEEP):
    """마스크 면적 비율 + T2 표준편차 기준으로 유효 슬라이스 선별 (최소 3장 보장)"""
    mask_areas = np.array([mask[s].sum() for s in range(patch.shape[0])])
    t2_stds   = np.array([patch[s, 1].std() for s in range(patch.shape[0])])
    keep = (mask_areas >= mask_areas.max()*min_mask_ratio) & (t2_stds >= min_std)
    keep_idx = np.where(keep)[0]
    # 유효 슬라이스가 min_keep 미만이면 면적 상위로 보충
    ...
    return patch[keep_idx], mask[keep_idx], info

# === 처리 흐름: 크롭 → dilation → per-slice 정규화 → 배경마스킹 → 슬라이스 필터링 → 저장 ===
# 출력: patches/*.npy (n_slices, 2[T1·T2], 128, 128), masks/*.npy, metadata.csv