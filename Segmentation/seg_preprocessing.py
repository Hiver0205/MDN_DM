# === 핵심 파라미터 ===
TARGET_SIZE = (512, 512)
TARGET_SPACING = 0.65        # mm/pixel — 등방성 목표 해상도
MIN_STRUCTURE_PIXELS = 100   # 슬라이스 유지 최소 구조물 픽셀 수
NUM_CLASSES = 9              # 배경·척추체·척추관 + IVD 6개

def remap_labels(mask_vol):
    """원본 라벨 → 9클래스 리매핑 (척추체 1~25→1, canal 100→2, IVD 201~206→3~8)"""
    new_mask = np.zeros_like(mask_vol, dtype=np.int8)
    for v in range(1, 26):
        new_mask[mask_vol == v] = 1
    new_mask[mask_vol == 100] = 2
    for i in range(1, 7):
        new_mask[mask_vol == (200 + i)] = i + 2
    return new_mask

def detect_sagittal_axis(sitk_image):
    """최대 spacing 축 = sagittal"""
    sp = sitk_image.GetSpacing()
    sag_axis_sp = max(range(3), key=lambda i: sp[i])
    return {0: 2, 1: 1, 2: 0}[sag_axis_sp]

def resample_t1_to_t2(t1_sitk, t2_sitk):
    """T1을 T2 그리드에 리샘플링 (두 시퀀스 공간 정합)"""
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(t2_sitk); r.SetInterpolator(sitk.sitkLinear); r.SetDefaultPixelValue(0)
    return r.Execute(t1_sitk)

def resample_slice_isotropic(img, sp, target):
    """2D 슬라이스 등방성 리샘플 (이미지: anti-aliasing + linear)"""
    zr, zc = sp[0]/target, sp[1]/target
    sigma = [max(0.0, 0.5/z - 0.5) if z < 1 else 0.0 for z in (zr, zc)]
    if any(s > 0 for s in sigma):
        img = gaussian_filter(img, sigma=sigma)
    return scipy_zoom(img, (zr, zc), order=1)

def pad_or_crop(img, th, tw):
    """중앙 기준 512×512 pad/crop (형태 왜곡 없이 크기 통일)"""
    h, w = img.shape
    # (상하·좌우 중앙 정렬 후 부족분 0 패딩 / 초과분 중앙 crop)
    ...

def preprocess_patient(patient_id):
    """T1·T2·mask 로드 → T1을 T2에 정합 → foreground z-score(각각) →
       라벨 리매핑 → sagittal 슬라이스 필터링 → isotropic 리샘플 → 512×512 저장"""
    t1 = sitk.ReadImage(...); t2 = sitk.ReadImage(...); mask = sitk.ReadImage(...)
    if t1.GetSize() != t2.GetSize() or t1.GetSpacing() != t2.GetSpacing():
        t1 = resample_t1_to_t2(t1, t2)
    # foreground(mask>0) 기준 z-score 정규화
    fg = mask_vol > 0
    t1_vol = (t1_vol - t1_vol[fg].mean()) / (t1_vol[fg].std() + 1e-8)
    t2_vol = (t2_vol - t2_vol[fg].mean()) / (t2_vol[fg].std() + 1e-8)
    mask_vol = remap_labels(mask_vol)
    # 구조물 < 100px 슬라이스 제거 후 isotropic 리샘플 + pad/crop
    ...

# === 클래스 가중치 (inverse frequency) ===
freq = class_pixels / class_pixels.sum()
weights = 1.0 / (freq + 1e-8)
weights = weights / weights.sum() * NUM_CLASSES   # IVD 등 희소 클래스 가중