"""
디텍션 + 깊이 + 로그 + GUI + 모듈화 + 예외처리
"""

import sys
import os
import cv2
from PyQt6 import uic, QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtCore import Qt
from util import DetectionThread

# 하이퍼 파라미터
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45
LOG_INTERVAL = 3    # 로그 간격(sec)
MODEL_PATH = "./model/yolo11n_rk3588.rknn"
CALIB_PATH = "./calib/stereo_calib.npz"
UI_PATH = "./ui/mainwindow.ui"
COLOR_BOX = (0, 255, 0)    # 녹색
COLOR_TEXT = (0, 255, 0)    # 녹색

# GUI
class DetectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        if not os.path.exists(UI_PATH):
            self.show_error_and_exit(f"Error: UI file not found at '{UI_PATH}'")
            return

        try:
            uic.loadUi(UI_PATH, self)
            self.setWindowTitle("KOREACORONA")
            self.showMaximized()

            if hasattr(self, "log_output"):
                self.log_output.setMaximumBlockCount(100)
            self.thread = DetectionThread(log_interval=LOG_INTERVAL, 
                                          model_path=MODEL_PATH, 
                                          calib_path=CALIB_PATH,
                                          camera_index=CAMERA_INDEX, 
                                          frame_width=FRAME_WIDTH, 
                                          frame_height=FRAME_HEIGHT, 
                                          input_size=INPUT_SIZE, 
                                          conf_threshold=CONF_THRESHOLD, 
                                          nms_threshold=NMS_THRESHOLD,
                                          color_box=COLOR_BOX, 
                                          color_text=COLOR_TEXT)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.log_signal.connect(self.update_log)
            self.thread.start()

        except Exception as e:
            self.show_error_and_exit(f"Initialization Error: {e}")

    def show_error_and_exit(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setText("Critical Error")
        msg_box.setInformativeText(message)
        msg_box.setWindowTitle("Error")
        msg_box.exec()
        sys.exit(1)

    def closeEvent(self, event):
        if hasattr(self, "thread"):
            self.thread.stop()
        event.accept()

    def update_log(self, msg):
        if hasattr(self, "log_output"):
            self.log_output.appendPlainText(msg)

    def update_image(self, cv_img):
        try:
            if cv_img is None or cv_img.size == 0:
                return
            
            qt_img = self.convert_cv_qt(cv_img)

            if hasattr(self, "video_display"):
                pixmap = qt_img.scaled(self.video_display.size(), 
                                    Qt.AspectRatioMode.KeepAspectRatio, 
                                    Qt.TransformationMode.SmoothTransformation)
                self.video_display.setPixmap(pixmap)
        
        except Exception as e:
            print(f"Image Update Error: {e}")

    def convert_cv_qt(self, cv_img):
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            return QtGui.QPixmap.fromImage(convert_to_Qt_format)
        
        except Exception as e:
            return QtGui.QPixmap()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    def exception_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = exception_hook
   
    window = DetectionWindow()
    window.show()
    sys.exit(app.exec())