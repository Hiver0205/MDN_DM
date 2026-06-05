# ── pipeline.py : 모델 정의 + 로딩 + 전처리 + 분할/분류 ──
IVD_START = 3                          # remap된 IVD 라벨 3~8
IVD_NAMES = {1:"L1-L2", 2:"L2-L3", 3:"L3-L4", 4:"L4-L5", 5:"L5-S1", 6:"S1-S2"}

# 학습 코드와 동일한 4개 모델 클래스를 추론용으로 재정의:
#   Spine25DCNN(팽윤), EndplateNetV9(종판), DiscModel(협착), smp.Unet(분할)

class SpinePipeline:
    def __init__(self, model_dir, device="cuda"):
        self.model_dir = model_dir; self.device = device
        self._load_models()

    def _load_models(self):
        """앱 시작 시 4개 모델 가중치를 메모리에 로드"""
        self.seg_model = smp.Unet("resnet34", in_channels=3, classes=9,
                                  decoder_attention_type="scse").to(self.device)
        self.seg_model.load_state_dict(torch.load(".../seg/best_unet.pth")["model_state_dict"])
        self.bulging_model   = Spine25DCNN(...);   # cla/bulging/best_model.pt
        self.endplate_model  = EndplateNetV9(...); # cla/endplate/model_fold2.pth
        self.narrowing_model = DiscModel(...);     # cla/narrowing/best_model.pth
        # 모두 .eval()

    def _preprocess_mri(self, t1_path, t2_path):
        """T1 로드 → T2를 T1 공간으로 정합(_resample_to_reference) → 각각 z-score"""
        ...
        return {"t1_sitk":..., "t1_vol_normed":..., "t2_vol_normed":...,
                "t1_vol_for_cla":..., "t2_vol_for_cla":...}

    @staticmethod
    def _simple_zscore(vol):
        """전경(>5퍼센타일) 기준 z-score 정규화"""
        ...

    # 분할: 시상면 슬라이스별 등방 리샘플 → 512 pad/crop → 3ch[T1,T2,avg] → U-Net
    def _run_segmentation(self, t1_normed, t2_normed, t1_sitk): ...
    # 디스크별 패치 추출 + 학습과 동일한 전처리(크롭·정규화·EP마스크) 후 분류
    def _classify_bulging(self, t1, t2, _m, t1_sitk, ivd_idx)  -> float
    def _classify_endplate(self, t1, t2, _m, t1_sitk, ivd_idx) -> (upper, lower)

# ── pipeline_ext.py : 4가지 개선을 monkey-patch ──
OPTIMAL_THRESHOLDS = {"bulging":0.40, "ep_upper":0.63, "ep_lower":0.58, "narrowing":0.52}

def _run_segmentation_with_tta(self, t1_normed, t2_normed, t1_sitk):
    """슬라이스별 U-Net 추론 + H-flip TTA(logits 평균) → argmax → 후처리"""
    ...   # _postprocess_seg_mask: IVD별 closing/opening + 최대 연결성분만 유지

def _classify_narrowing(self, t1, t2, seg_mask, t1_sitk, ivd_idx, tta_rounds=10):
    """협착 패치 추출 후 10회 TTA 평균 확률"""
    ...

def run(self, t1_path, t2_path, tta_rounds=10):
    """전체 파이프라인 단일 진입점"""
    preproc = self._preprocess_mri(t1_path, t2_path)
    seg_mask_3d = _run_segmentation_with_tta(self, preproc["t1_vol_normed"],
                                             preproc["t2_vol_normed"], preproc["t1_sitk"])
    present_ivds = [i for i in range(1,7) if np.any(seg_mask_3d == IVD_START+i-1)]
    diagnoses = {}
    for ivd_idx in present_ivds:                       # 검출된 디스크별 반복
        b      = self._classify_bulging(...)
        eu, el = self._classify_endplate(...)
        n      = self._classify_narrowing(..., tta_rounds=tta_rounds)
        diagnoses[IVD_NAMES[ivd_idx]] = {"bulging": b,
                                         "endplate": {"upper": eu, "lower": el},
                                         "narrowing": n}
    return {"seg_mask": seg_mask_3d, "diagnoses": diagnoses,
            "thresholds": OPTIMAL_THRESHOLDS}

def patch_pipeline():
    """SpinePipeline에 개선 메서드(run·_classify_narrowing 등) 주입"""
    SpinePipeline._classify_narrowing = _classify_narrowing
    SpinePipeline.run = run
    ...