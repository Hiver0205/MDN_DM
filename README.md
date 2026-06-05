# MDN (MRI-Disc Net)

> **T1·T2 MRI로 허리 디스크를 자동 분석하는 멀티모달 진단 보조 시스템**
> A multimodal lumbar disc diagnostic assistant based on the SPIDER dataset.

---

## 프로젝트 소개

허리 디스크 진단은 MRI를 보는 의료진의 경험에 크게 의존하고, 같은 영상도 보는 사람에 따라 판단이 달라질 수 있습니다. **MDN**은 이 과정에 객관적이고 정량적인 기준을 더하기 위해 만들어졌습니다.

환자의 T1·T2 MRI를 넣으면, AI가 추간판(IVD)을 자동으로 찾아 분할하고 디스크마다 **팽윤·종판 변성·협착** 가능성을 수치로 알려줍니다. 결과는 표뿐 아니라 회전·확대가 가능한 **3D 모델**로 보여주어, 어느 디스크에 어떤 문제가 있는지 한눈에 파악할 수 있습니다.

별도 서버나 설치 과정 없이 **실행 파일 하나로 동작하는 데스크톱 프로그램**으로, 의료진이 MRI만 올리면 바로 분석 결과를 확인할 수 있도록 설계했습니다.

---

## 이런 걸 할 수 있어요

- 📂 **MRI 업로드** — DICOM 폴더 또는 MHA 파일을 올리면 됩니다 (DICOM은 T1/T2 자동 구분·변환)
- 🧠 **자동 분석** — 디스크 분할부터 병변 분류까지 한 번에 수행
- 🎨 **3D 시각화** — 척추·디스크를 병변 색상과 함께 입체적으로 확인, 디스크·병변별 필터 제공
- 📊 **정량 결과** — 디스크별 병변 확률과 양성/음성 판정을 카드·테이블로 제공
- 💾 **결과 저장** — 분석 결과를 CSV로 내보내기

---

## 어떻게 동작하나요

```
MRI 입력 → 전처리(정합·정규화) → 디스크 분할(U-Net) → 병변 분류(팽윤·종판·협착) → 3D 시각화 + 결과 표
```

내부적으로는 4개의 딥러닝 모델이 하나의 파이프라인으로 연결되어 있습니다.

| 단계 | 모델 | 핵심 |
|---|---|---|
| 디스크 분할 | U-Net (ResNet34) | 척추체·척추관·6개 IVD를 9클래스로 분할 |
| 팽윤 | ResNet18 + Soft MaskedGAP | 디스크별 팽윤 여부 |
| 종판 | EfficientNet-B0 + Multi-Scale SE | 상·하 종판 변성 |
| 협착 | 경량 CNN + Attention Pooling | 디스크별 협착 여부 |

분할 IVD Dice **0.84**, 협착 분류 AUC **0.95** 등, 주요 지표에서 안정적인 성능을 보입니다.

---

## 폴더 구성

```
MDN/
├── seg/            # 디스크 분할 (전처리 + 학습)
├── cla/            # 병변 분류 — bulging / endplate / narrowing
└── Discprogram/    # 데스크톱 앱 (PyQt5 GUI · 파이프라인 · 3D 렌더 · DICOM 변환)
```

---

## 실행 방법

```bash
pip install torch torchvision segmentation_models_pytorch timm albumentations \
            SimpleITK pydicom numpy pandas scipy scikit-learn scikit-image \
            pyvista pyvistaqt PyQt5 matplotlib tqdm

cd Discprogram
python app_desktop.py
```

> Windows 10/11 · Python 3.10+ (GPU 권장). 단독 실행형 `.exe`로 빌드하면 Python 설치 없이 바로 실행됩니다.

---

## 데이터셋

[SPIDER](https://spider.grand-challenge.org/) — 척추 T1·T2 MRI와 분할 마스크, 방사선학적 등급 데이터를 사용합니다.

---

> ⚠️ 본 시스템은 연구·학습 목적의 **진단 보조 도구**이며, 실제 임상 진단을 대체하지 않습니다.
