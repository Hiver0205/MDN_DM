import pyvista as pv
from skimage import measure

IVD_LABELS = {201:"L1-L2", 202:"L2-L3", 203:"L3-L4", 204:"L4-L5", 205:"L5-S1", 206:"S1-S2"}
BONE_MIN, BONE_MAX = 1, 99
# 병변 색상: 정상 파랑 / 협착 빨강 / 팽윤 주황 / 종판 보라 / 진단없음 회색
IVD_NORMAL_COLOR, IVD_NARROW_COLOR = "#2196F3", "#E53935"
IVD_BULGE_COLOR, IVD_ENDPLATE_COLOR, IVD_NODIAG_COLOR = "#FF9800", "#9C27B0", "#6E6E6E"
BONE_OPACITY, IVD_DIM_OPACITY = 0.12, 0.55      # 뼈 반투명 / 미선택 디스크 흐리게

def _label_to_mesh(binary_zyx, spacing_zyx, step_size=1, smooth_iter=20):
    """이진 볼륨 → marching_cubes → PolyData 메시 → Laplacian smooth"""
    verts, faces, _, _ = measure.marching_cubes(binary_zyx.astype(np.float32),
                                                level=0.5, spacing=spacing_zyx, step_size=step_size)
    pvf = np.empty((faces.shape[0], 4), np.int64); pvf[:,0] = 3; pvf[:,1:] = faces
    mesh = pv.PolyData(verts, pvf.ravel())
    return mesh.smooth(n_iter=smooth_iter, relaxation_factor=0.1) if smooth_iter else mesh

def _ivd_color(disc_name, diag, th, lesion_filter="all"):
    """진단 결과 + 임계값으로 디스크 색 결정 (필터: all/narrowing/bulging/endplate)"""
    s = diag[disc_name]
    narrow = s["narrowing"] >= th["narrowing"]
    bulge  = s["bulging"]   >= th["bulging"]
    ep     = (s["endplate"]["upper"] >= th["ep_upper"]) or (s["endplate"]["lower"] >= th["ep_lower"])
    if lesion_filter == "narrowing": return IVD_NARROW_COLOR if narrow else IVD_NODIAG_COLOR
    if lesion_filter == "bulging":   return IVD_BULGE_COLOR  if bulge  else IVD_NODIAG_COLOR
    if lesion_filter == "endplate":  return IVD_ENDPLATE_COLOR if ep   else IVD_NODIAG_COLOR
    # all: 우선순위 협착 > 팽윤 > 종판 > 정상
    if narrow: return IVD_NARROW_COLOR
    if bulge:  return IVD_BULGE_COLOR
    if ep:     return IVD_ENDPLATE_COLOR
    return IVD_NORMAL_COLOR

def _set_camera(pl, view="lateral", lr_axis=None):
    """spacing 최대 축(좌우)을 따라 보는 측면(시상면) 뷰, up=(1,0,0)"""
    ...

def _build_plotter(mask_path, diag=None, thresholds=None, lesion_filter="all", highlight_disc=None):
    """뼈(반투명) + IVD 메시(병변 색상)를 Plotter에 올림. 선택 디스크는 선명, 나머지 흐리게"""
    arr, spacing_zyx, _ = _load_mask(mask_path)
    pl = pv.Plotter(off_screen=True); pl.set_background("#0d0d0f")
    pl.add_mesh(_get_bone_mesh(arr, spacing_zyx), color=BONE_COLOR, opacity=BONE_OPACITY)
    for lbl in IVD_LABELS:                          # 201~206 각 IVD 별도 메시
        if (arr == lbl).sum() == 0: continue
        mesh = _label_to_mesh((arr == lbl).astype(np.uint8), spacing_zyx, smooth_iter=60)
        color = _ivd_color(IVD_LABELS[lbl], diag, thresholds, lesion_filter)
        op = IVD_OPACITY if (highlight_disc in (None, IVD_LABELS[lbl])) else IVD_DIM_OPACITY
        pl.add_mesh(mesh, color=color, opacity=op, smooth_shading=True, ...)
    _set_camera(pl, view="lateral", lr_axis=int(np.argmax(spacing_zyx)))
    return pl

def count_discs(mask_path):
    """마스크에 존재하는 IVD(201~206) 이름 리스트 (0개 환자 대비)"""
    arr, _, _ = _load_mask(mask_path)
    return [IVD_LABELS[l] for l in IVD_LABELS if np.any(arr == l)]

# render_html(): export_html로 마우스 회전 HTML 생성 / render_png(): 정적 PNG (폴백용)