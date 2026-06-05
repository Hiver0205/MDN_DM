# === 핵심 파라미터 ===
PATCH_SIZE = 128
TARGET_SAG_SPACING, MIN_SAG_SLICES, MAX_SLICES = 3.3, 3, 24
EP_RATIO   = 0.2     # EP mask: IVD 높이의 상·하 20%씩 (겹침 방지)
MARGIN_MULT, MIN_CROP = 0.2, 64
LOW_STD_THRESH = 0.01
IVD_MIN, IVD_MAX = 1, 6
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# 라벨: SPIDER radiological_gradings.csv의 UP/LOW endplate를 >0 → 1로 이진화
df["upper_ep"] = (df["upper_ep_raw"] > 0).astype(int)
df["lower_ep"] = (df["lower_ep_raw"] > 0).astype(int)
# 환자 단위 stratified train/val/test 분할 (양성=upper+lower>0 기준)

def resample_to_reference(moving_img, reference_img):
    """T2를 T1 공간으로 resample (SimpleITK, Linear)"""
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(reference_img); r.SetInterpolator(sitk.sitkLinear); r.SetDefaultPixelValue(0)
    return sitk.GetArrayFromImage(r.Execute(moving_img))

def foreground_zscore(volume, mask):
    """전경(mask>0) 기준 z-score 정규화"""
    fg = mask > 0
    mu, sigma = volume[fg].mean(), volume[fg].std()
    return ((volume - mu) / sigma).astype(np.float32) if sigma >= 1e-6 else np.zeros_like(volume, np.float32)

def extract_one_ivd(pid, ivd_label):
    """IVD 1개 추출: T2→T1 정합 + z-score → IVD bbox와 인접 척추체 중심으로 z범위 크롭
       → sagittal 서브샘플링(3.3mm, 3~24장) → anti-aliasing resize 128 → (N,2,128,128)"""
    ivd_code = 200 + ivd_label
    t1_vol, t1_sitk = load_mha(...); t1_mask, _ = load_mha(...); t2_vol, t2_sitk = load_mha(...)
    if not np.any(t1_mask == ivd_code): return None
    t2_resampled = resample_to_reference(t2_sitk, t1_sitk)
    t1_norm = foreground_zscore(t1_vol, t1_mask)
    t2_norm = foreground_zscore(t2_resampled, t1_mask)
    # IVD bbox + 상하 vertebra centroid 기반 z-margin, y는 높이의 0.5 margin으로 크롭
    ...
    return {"patches": patches, "ivd_mask": ivd_mask, "n_sag": n_slices}

def compute_endplate_mask(ivd_mask, ep_ratio=EP_RATIO):
    """비율 기반 EP mask 생성 — 각 x좌표에서 IVD 높이의 상·하 ep_ratio(20%)를 종판으로,
       두께는 IVD높이/2 이하로 제한해 상하 띠 겹침(떡짐) 방지 → EP < IVD 보장"""
    ep_mask = np.zeros_like(ivd_mask, dtype=np.float32)
    for s in range(ivd_mask.shape[0]):
        for x in range(ivd_mask.shape[2]):
            ys = np.where(ivd_mask[s, :, x] > 0.5)[0]
            if len(ys) < 2: continue
            top, bot = ys.min(), ys.max()
            thick = min(max(1, round((bot-top+1)*ep_ratio)), (bot-top+1)//2)
            ep_mask[s, top:top+thick, x] = 1.0          # 상단 종판
            ep_mask[s, bot-thick+1:bot+1, x] = 1.0      # 하단 종판
    return ep_mask

def crop_endplate_center(patches, ivd_mask, ep_mask):
    """종판 중심으로 타이트 크롭(마진 0.2, 최소 64) 후 128로 anti-aliasing 리사이즈"""
    ...

def filter_low_std_slices(patches, ivd_mask, ep_mask, threshold=LOW_STD_THRESH):
    """STD가 낮은(거의 빈) 슬라이스 제거, 최소 3장 보장"""
    keep = [s for s in range(patches.shape[0]) if max(patches[s,0].std(), patches[s,1].std()) > threshold]
    if len(keep) < MIN_SAG_SLICES: return patches, ivd_mask, ep_mask
    keep = np.array(keep)
    return patches[keep], ivd_mask[keep], ep_mask[keep]

# === 처리 흐름: IVD 추출(정합·정규화) → EP mask 생성 → EP 중심 크롭 → low-STD 필터 → 저장 ===
# 출력: patch_t1/, patch_t2/, ivdmask/, epmask/ (각 128×128), metadata_v9.csv