# 모델 정의 트리거 전에 파이프라인 확장 패치 적용
import pipeline_ext; pipeline_ext.patch_pipeline()
from pipeline import SpinePipeline

# OpenGL/3D 가용성 확인 (없으면 3D 비활성화)
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_3D = True
except Exception:
    HAS_3D = False

# 앱 판정 임계값 (pipeline_ext와 동일)
THRESHOLDS = {"bulging":0.40, "ep_upper":0.63, "ep_lower":0.58, "narrowing":0.52}

# === 비동기 워커 (UI 비차단) ===
class PipelineWorker(QThread):
    progress = pyqtSignal(str); finished = pyqtSignal(dict); error = pyqtSignal(str)
    def run(self):
        try:    self.finished.emit(self.pipe.run(self.t1_path, self.t2_path))
        except Exception as e: self.error.emit(...)

class DicomWorker(QThread):
    progress = pyqtSignal(str); finished = pyqtSignal(dict); error = pyqtSignal(str)
    def run(self):
        out_dir = tempfile.mkdtemp(prefix="disc_mha_")
        result = dicom_to_mha.convert_dicom_folder(self.dicom_dir, out_dir, patient_tag="upload")
        self.finished.emit(result)

# === 진단 카드 (병변별 확률·양성/음성 표시) ===
class DiagCard(QFrame):
    def update_data(self, prob, threshold):
        positive = prob >= threshold      # 임계값 기준 양성/음성 + 색상

# === 메인 윈도우 ===
class MainWindow(QMainWindow):
    def _init_pipeline(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = SpinePipeline(_SCRIPT_DIR, device=device)   # 모델 4종 로딩

    def _on_analyze(self):
        # 입력 모드(DICOM/MHA) 검증 후 DICOM이면 변환 → 파이프라인 실행
        if self.radio_dicom.isChecked(): self._start_dicom_conversion()
        else: self._start_pipeline(self.t1_path, self.t2_path)

    def _on_dicom_done(self, result):        # 변환된 T1/T2로 파이프라인 시작
        self._start_pipeline(result["t1"], result["t2"])

    def _start_pipeline(self, t1, t2):
        self.worker = PipelineWorker(self._pipe, t1, t2)
        self.worker.finished.connect(self._on_pipeline_done); self.worker.start()

    def _on_pipeline_done(self, results):    # 결과 수신 → 시각화·테이블 채움
        self._diag = results["diagnoses"]
        self._seg_mask = _convert_seg_mask(results["seg_mask"])   # 라벨→201~206
        self._render_3d(); self._populate_table(); self.btn_export_csv.setEnabled(True)

    def _render_3d(self):                    # QtInteractor에 메시 추가 (1회 생성)
        # 뼈(반투명) + IVD별 메시(marching_cubes), 측면 카메라
        ...
    def _apply_appearance(self):             # 재렌더 없이 병변 필터·선택 디스크 색만 갱신
        base_color = _ivd_color(disc_name, self._diag, THRESHOLDS, filt)
    def _update_right_panel(self, disc_name):# 진단 카드 4개 + 요약 문구 갱신
        ...
    def _populate_table(self):               # 전체 디스크 × 병변별 확률·판정 테이블
        ...
    def _export_csv(self):                    # 결과 CSV 저장 (utf-8-sig)
        path, _ = QFileDialog.getSaveFileName(self, "CSV 저장", "diagnosis_results.csv", ...)
        writer = csv.writer(f)
        writer.writerow(["디스크","Bulging","EP_Upper","EP_Lower","Narrowing","판정"])
        ...

if __name__ == "__main__":
    app = QApplication(sys.argv); win = MainWindow(); win.show(); sys.exit(app.exec_())