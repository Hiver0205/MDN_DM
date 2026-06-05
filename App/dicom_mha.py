# SimpleITK 필수, pydicom 있으면 더 정확한 메타데이터 분류
import SimpleITK as sitk
try: import pydicom; _HAS_PYDICOM = True
except Exception: _HAS_PYDICOM = False

def _scan_series(dicom_dir):
    """하위 폴더까지 재귀 탐색하여 모든 DICOM 시리즈를 찾는다 (GetGDCMSeriesIDs)"""
    series_list, seen_uid = [], set()
    candidate_dirs = {dicom_dir} | {root for root, _, _ in os.walk(dicom_dir)}
    reader = sitk.ImageSeriesReader()
    for d in sorted(candidate_dirs):
        for sid in reader.GetGDCMSeriesIDs(d):
            if sid in seen_uid: continue
            files = reader.GetGDCMSeriesFileNames(d, sid)
            seen_uid.add(sid)
            info = _read_series_meta(files, sid)   # desc, scan_seq, TR/TE/TI 추출
            info.update(dir=d, files=files, n=len(files))
            series_list.append(info)
    return series_list

def _classify_weighting(info):
    """시리즈를 T1/T2/other로 추정: ① 텍스트 키워드 → ② TR/TE 물리 추정"""
    blob = info["blob"]
    # 요추 디스크에 안 쓰는 시퀀스 제외
    for kw in ["stir","flair","dwi","diffusion","adc","myelo","localizer",
               "scout","survey","calibration","space_t2","t2_space"]:
        if kw in blob: return ("other", 0.0, f"excluded '{kw}'")
    # 명시적 키워드
    has_t1 = bool(re.search(r"\bt1\b|t1w|t1_|t1-", blob))
    has_t2 = bool(re.search(r"\bt2\b|t2w|t2_|t2-", blob))
    if has_t1 and not has_t2: return ("t1", 0.95, "desc T1")
    if has_t2 and not has_t1: return ("t2", 0.95, "desc T2")
    # TR/TE 물리 추정 (T1: TR<900 & TE<30 / T2: TR≥1800 & TE≥50)
    TR, TE = info["TR"], info["TE"]
    if TR and TE:
        if TR < 900 and TE < 30:   return ("t1", 0.8, "TR/TE → T1")
        if TR >= 1800 and TE >= 50: return ("t2", 0.8, "TR/TE → T2")
        if TE >= 60: return ("t2", 0.6, "long TE → T2")
        if TE < 20:  return ("t1", 0.6, "short TE → T1")
    if "IR" in info["scan_seq"]: return ("t1", 0.55, "IR → T1")
    return ("other", 0.0, "unclassified")

def _pick_best(cands):
    """후보가 여러 개면 (분류 score → 슬라이스 수 → sagittal 우선)으로 1개 선택"""
    return max(cands, key=lambda c: (round(c["score"],2), c["info"]["n"],
                                     1 if "sag" in c["info"]["blob"] else 0)) if cands else None

def convert_dicom_folder(dicom_dir, out_dir, patient_tag="patient", verbose=True):
    """환자 DICOM 폴더 → T1/T2 .mha 변환 (단일 진입점)"""
    series_list = _scan_series(dicom_dir)
    if not series_list:
        raise RuntimeError("DICOM 시리즈를 찾지 못했습니다.")
    cls = classify_series(series_list)            # T1/T2 후보 분류 후 best 선택
    result = {"t1": None, "t2": None, "report": [], "n_series": len(series_list)}
    for w in ("t1", "t2"):                        # 선택된 시리즈를 볼륨으로 읽어 .mha 저장
        if cls[w] is not None:
            img = _read_series_volume(cls[w]["files"])     # ImageSeriesReader
            path = os.path.join(out_dir, f"{patient_tag}_{w}.mha")
            sitk.WriteImage(img, path, useCompression=True)
            result[w] = path
    return result   # {"t1": 경로|None, "t2": 경로|None, "report": [...], "n_series": N}