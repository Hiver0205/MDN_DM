# === 핵심 파라미터 ===
PATCH_SIZE = TARGET_CROP = 128
MARGIN_MULT = 0.05          # 디스크 재크롭 마진 (협착 모듈 고유값)
MIN_CROP, MIN_SLICES, MAX_SLICES = 40, 3, 24
TARGET_SAG_SPACING, LOW_STD = 3.3, 0.01

def resample(mv, ref, mask=False):
    """T2→T1 정합 (마스크는 NearestNeighbor, 영상은 Linear)"""
    r = sitk.ResampleImageFilter(); r.SetReferenceImage(ref)
    r.SetInterpolator(sitk.sitkNearestNeighbor if mask else sitk.sitkLinear)
    r.SetDefaultPixelValue(0); return r.Execute(mv)

def fg_zscore(img, msk):
    """전경(mask>0) 기준 z-score 정규화"""
    fg = img[msk > 0].astype(np.float32)
    if len(fg) == 0 or fg.std() < 1e-6: return img.astype(np.float32)
    return (img.astype(np.float32) - fg.mean()) / fg.std()

def load_patient(pid):
    """T1·T2·mask(.mha) 로드 → T2를 T1에 정합 → T1·T2 각각 foreground z-score"""
    t1s, t2s, ms = load_sitk(...), load_sitk(...), load_sitk(...)
    t2r = resample(t2s, t1s)
    t1a, t2a, ma = sitk_to_np(t1s), sitk_to_np(t2r), sitk_to_np(ms)
    return fg_zscore(t1a, ma), fg_zscore(t2a, ma), ma, t1s.GetSpacing()

def sel_sag(m3d, ivd, spx):
    """해당 IVD(코드 200+ivd)가 있는 sagittal 슬라이스 선택 (3.3mm 서브샘플, 3~24장)"""
    ...

def extract_s1(t1, t2, msk, sp, ivd):
    """IVD bbox + 인접 척추체 z-margin + y는 높이/2 margin으로 크롭 → 128 리사이즈
       → low-STD 슬라이스 제거 (최소 3장 보장)"""
    ...
    return patches, mask_out, None      # (N, 2[T1·T2], 128, 128)

def recrop(patches, masks, ivd):
    """디스크 union bbox 중심으로 마진 0.05 재크롭 → 128 리사이즈"""
    du = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool)
    for s in range(patches.shape[0]): du |= (masks[s] == 200+ivd)
    ys, xs = np.where(du)
    dm = max(ys.ptp()+1, xs.ptp()+1)
    crop = max(MIN_CROP, min(dm + 2*int(dm*MARGIN_MULT), PATCH_SIZE))   # 마진 0.05
    # 중심 기준 정사각형 crop 후 TARGET_CROP(128)로 zoom
    ...
    return out_patch, out_mask, crop

# === 처리 흐름 (metadata.csv의 split 따라):
#   환자 로드(정합·정규화) → sagittal 선택 → IVD 크롭(extract_s1) → 디스크 중심 재크롭(recrop)
#   → 빈 패치 제외 → train/val/test 폴더에 저장 ===
# 출력: patches/{train,val,test}/{pid}_ivd{n}.npy + _mask.npy, preprocess_report.json