
import sys
import cv2
import numpy as np
import requests
from PyQt6 import uic, QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import os

# --- Parameters ---
UI_PATH = "./ui/mainwindow.ui"
BACKEND_URL = "http://127.0.0.1:8000/video_feed" # OrangePi의 IP 주소로 변경해야 합니다.

class VideoThread(QThread):
    """ continuously receives video frames from backend """
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, url):
        super().__init__()
        self._run_flag = True
        self.url = url

    def run(self):
        try:
            # Use stream=True to handle the streaming response
            with requests.get(self.url, stream=True, timeout=5) as r:
                r.raise_for_status() # Raise an exception for bad status codes
                
                byte_buffer = b''
                for chunk in r.iter_content(chunk_size=1024*10):
                    if not self._run_flag:
                        break
                        
                    byte_buffer += chunk
                    # Look for the start and end of a JPEG image
                    start = byte_buffer.find(b'\xff\xd8')
                    end = byte_buffer.find(b'\xff\xd9')
                    
                    if start != -1 and end != -1:
                        jpg = byte_buffer[start:end+2]
                        byte_buffer = byte_buffer[end+2:]
                        
                        try:
                            # Decode the JPEG image to a numpy array (cv2 image)
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                self.change_pixmap_signal.emit(frame)
                        except Exception as e:
                            print(f"Frame decode error: {e}")

        except requests.exceptions.RequestException as e:
            print(f"Could not connect to backend: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
        print("VideoThread finished.")

    def stop(self):
        """Sets a flag to stop the thread."""
        self._run_flag = False
        self.wait()

class ClientWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        if not os.path.exists(UI_PATH):
            print(f"Error: UI file not found at '{UI_PATH}'")
            sys.exit(1)

        uic.loadUi(UI_PATH, self)
        self.setWindowTitle("Detection Client")
        self.showMaximized()

        # The log_output might not be used anymore if logs are on the server
        # but we keep it to avoid errors with the UI file.
        if hasattr(self, "log_output"):
            self.log_output.setMaximumBlockCount(100)
            self.log_output.appendPlainText("Client started. Connecting to backend...")

        # Create the video thread
        self.thread = VideoThread(BACKEND_URL)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def closeEvent(self, event):
        print("Closing window...")
        self.thread.stop()
        event.accept()

    def update_image(self, cv_img):
        """Updates the video_display label with a new opencv image."""
        qt_img = self.convert_cv_qt(cv_img)
        if hasattr(self, "video_display"):
            # Scale the image to fit the label, keeping aspect ratio
            pixmap = qt_img.scaled(
                self.video_display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_display.setPixmap(pixmap)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888
        )
        return QtGui.QPixmap.fromImage(convert_to_Qt_format)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClientWindow()
    window.show()
    sys.exit(app.exec())
