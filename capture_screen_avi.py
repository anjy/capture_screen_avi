"""
PyQt5로 "현재 화면(전체 데스크탑)"을 실시간 미리보기로 보이면서 동시에 동영상으로 저장하는 최소 예제.
- 미리보기: QLabel에 QPixmap 표시 (QScreen.grabWindow)
- 녹화: OpenCV VideoWriter로 mp4 파일 저장 (H.264/MP4V 중 환경 맞는 코덱 선택)

필요 패키지
pip install PyQt5 opencv-python

주의
- 고해상도/다중 모니터 환경에서는 FPS를 낮추거나 캡쳐 영역을 줄이세요.
- 오디오 녹음은 포함하지 않습니다. (필요 시 pyaudio/ffmpeg 파이프 조합 권장)


UI없이 한파일로 실행 하는 파일
"""

import sys
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QSpinBox,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QCheckBox,
    QComboBox,
)

import numpy as np
import cv2


def qimage_from_pixmap(pix: QPixmap) -> QImage:
    # QPixmap -> QImage (원본 비트맵 유지)
    img = pix.toImage()
    # for safety, ensure format is RGB32 (BGRA on little endian)
    return img.convertToFormat(QImage.Format.Format_RGB32)


def qimage_to_numpy(img: QImage) -> np.ndarray:
    # QImage (Format_RGB32) -> Numpy (H, W, 4) BGRA
    width = img.width()
    height = img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
    return arr


class ScreenRecorder(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("화면 미리보기 + 동영상 캡쳐")
        self.resize(980, 620)

        # UI
        self.preview = QLabel("미리보기 대기 중…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111; color:#bbb; border:1px solid #222;")
        self.preview.setMinimumSize(640, 360)

        self.btn_start = QPushButton("● 녹화 시작")
        self.btn_stop = QPushButton("■ 정지")
        self.btn_stop.setEnabled(False)

        self.fps_box = QSpinBox()
        self.fps_box.setRange(1, 60)
        self.fps_box.setValue(20)
        self.fps_box.setPrefix("FPS ")

        self.scale_box = QComboBox()
        self.scale_box.addItems(["원본", "1/2", "1/3", "1/4"])

        self.keep_ratio = QCheckBox("미리보기 비율 유지")
        self.keep_ratio.setChecked(True)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_start)
        top_bar.addWidget(self.btn_stop)
        top_bar.addStretch(1)
        top_bar.addWidget(self.fps_box)
        top_bar.addWidget(self.scale_box)
        top_bar.addWidget(self.keep_ratio)

        root = QVBoxLayout()
        root.addLayout(top_bar)
        root.addWidget(self.preview, 1)
        self.setLayout(root)

        # 상태
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.recording = False
        self.screen = QApplication.primaryScreen()
        self.last_frame_time = 0.0

        # Video
        self.writer = None
        self.target_path = None
        self.target_size = None  # (w,h)

        # 이벤트
        self.btn_start.clicked.connect(self._start_recording)
        self.btn_stop.clicked.connect(self._stop_recording)

    def _choose_target(self) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"screen_{ts}.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "저장 위치", default_name, "MP4 (*.mp4);;AVI (*.avi)")
        return Path(path) if path else None

    def _scale_factor(self) -> float:
        m = self.scale_box.currentText()
        return {"원본": 1.0, "1/2": 0.5, "1/3": 1/3, "1/4": 0.25}[m]

    def _init_writer(self, frame_w: int, frame_h: int, fps: int, out_path: Path):
        # FourCC: mp4v는 대부분의 Windows에서 동작. H.264('avc1')는 코덱 설치 필요할 수 있음.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") if out_path.suffix.lower() == ".mp4" else cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))
        if not writer.isOpened():
            raise RuntimeError("VideoWriter를 열 수 없습니다. 다른 코덱/확장자를 시도하세요.")
        return writer

    def _start_recording(self):
        if self.recording:
            return
        self.target_path = self._choose_target()
        if not self.target_path:
            return

        fps = self.fps_box.value()
        interval_ms = int(1000 / max(1, fps))
        self.timer.start(interval_ms)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.recording = True
        self.last_frame_time = 0.0

    def _stop_recording(self):
        if not self.recording:
            return
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.recording = False
        self._release_writer()
        self.target_path = None
        self.target_size = None

    def closeEvent(self, e):
        self._stop_recording()
        super().closeEvent(e)

    def _release_writer(self):
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None

    def _on_tick(self):
        if not self.screen:
            return
        # 1) 화면 캡쳐 (전체 데스크탑)
        pix = self.screen.grabWindow(0)
        # HiDPI 스케일 보정
        dpr = pix.devicePixelRatio()
        if dpr != 1.0:
            # QPixmap 내부 버퍼는 물리 픽셀 기준. 미리보기/저장 시 사용 크기 보정 필요.
            pix.setDevicePixelRatio(1.0)

        # 2) 미리보기 표시 (라벨 크기에 맞춤)
        if self.keep_ratio.isChecked():
            scaled = pix.scaled(self.preview.size() * self.devicePixelRatioF(), Qt.AspectRatioMode.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            scaled = pix.scaled(self.preview.size() * self.devicePixelRatioF(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

        if not self.recording:
            return

        # 3) 파일로 쓰기 (OpenCV)
        img = qimage_from_pixmap(pix)
        arr = qimage_to_numpy(img)  # BGRA

        # 다운스케일 옵션
        scale = self._scale_factor()
        if scale != 1.0:
            new_w = int(arr.shape[1] * scale)
            new_h = int(arr.shape[0] * scale)
            arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # BGRA -> BGR (OpenCV는 3채널 필요)
        frame_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        # writer 초기화 (첫 프레임에서)
        if self.writer is None:
            h, w = frame_bgr.shape[:2]
            fps = self.fps_box.value()
            self.writer = self._init_writer(w, h, fps, self.target_path)
            self.target_size = (w, h)

        # 크기 변동 방지
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != self.target_size:
            frame_bgr = cv2.resize(frame_bgr, self.target_size, interpolation=cv2.INTER_AREA)

        self.writer.write(frame_bgr)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ScreenRecorder()
    w.show()
    sys.exit(app.exec_())
