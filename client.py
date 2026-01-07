import sys
import cv2
import numpy as np
import requests
from PyQt6 import uic, QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import os

UI_PATH = "./ui/mainwindow.ui"
BACKEND_URL = "192.168.0.24"

# 스트리밍 클래스
class VideoThread(QThread):
    """
    백엔드 서버로부터 비디오 프레임 수신
    수신된 프레임을 PyQt 시그널을 통해 메인 윈도우로 전달
    QThread 비동기
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)
    connection_failed = pyqtSignal(str)

    # 쓰레드 초기화
    def __init__(self, url):
        super().__init__()
        self._run_flag = True  # 쓰레드 플래그
        self.url = url

    # 쓰레드 실행
    def run(self):
        try:
            # 5초 이상 응답이 없으면 예외처리.
            with requests.get(self.url, stream=True, timeout=5) as r:
                r.raise_for_status()

                # 바이널 버퍼
                byte_buffer = b''
                # 10KB 청크 10개
                for chunk in r.iter_content(chunk_size=1024 * 10):
                    if not self._run_flag:
                        break
    
                    byte_buffer += chunk
                    # JPEG 이미지의 시작, 끝 마커
                    start = byte_buffer.find(b'\xff\xd8')
                    end = byte_buffer.find(b'\xff\xd9')

                    # 시작과 끝 마커가 모두 있으면 실행
                    if start != -1 and end != -1:
                        # JPEG 추출
                        jpg = byte_buffer[start:end + 2]
                        byte_buffer = byte_buffer[end + 2:]

                        try:
                            # JPEG -> numpy array
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                self.change_pixmap_signal.emit(frame)
                        except Exception as e:
                            print(f"프레임 디코딩 오류: {e}")

        except requests.exceptions.RequestException as e:
            error_msg = f"서버에 연결할 수 없습니다.\n{e}"
            print(error_msg)
            self.connection_failed.emit(error_msg)
        except Exception as e:
            error_msg = f"예상치 못한 오류가 발생했습니다.\n{e}"
            print(error_msg)
            self.connection_failed.emit(error_msg)

        print("VideoThread가 종료되었습니다.")

    # 쓰레드 정지
    def stop(self):
        self._run_flag = False
        self.wait()

# GUI 출력
class ClientWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        if not os.path.exists(UI_PATH):
            print(f"오류: UI 파일을 다음 경로에서 찾을 수 없습니다 '{UI_PATH}'")
            sys.exit(1)

        uic.loadUi(UI_PATH, self)
        # 창 제목
        self.setWindowTitle("Client")
        # 전체화면
        self.showMaximized()

        self.thread = None

        # 요소들이 존재하는지 확인
        if hasattr(self, "log_output"):
            # 로그 출력창 줄 수 제한
            self.log_output.setMaximumBlockCount(100)
            self.log_output.appendPlainText("클라이언트가 시작되었습니다.\nIP를 입력하고 Connect 버튼을 누르세요.")

        # URL 필드
        if hasattr(self, "backend_url_input"):
            self.backend_url_input.setText(BACKEND_URL)

        # connect 버튼
        if hasattr(self, "connect_button"):
            self.connect_button.clicked.connect(self.connect_to_backend)

        # disconnect 버튼
        if hasattr(self, "disconnect_button"):
            self.disconnect_button.clicked.connect(self.disconnect_from_backend)
            # 초기값 비활성화
            self.disconnect_button.setEnabled(False)

    # 백엔드 연결
    def connect_to_backend(self):
        # 중복 연결 방지
        if self.thread is not None and self.thread.isRunning():
            self.log_output.appendPlainText("이미 연결되어 있습니다.")
            return

        address = self.backend_url_input.text().strip()
        url = address

        # ip만으로 접속하기위한 문자열 처리
        if "://" not in url:
            url = "http://" + url
        
        if ":" not in url.split('://')[1]:
            url += ":8000"
            
        self.log_output.appendPlainText(f"{url}에 연결 중...")

        # 비디오 스레드 생성
        self.thread = VideoThread(url)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.connection_failed.connect(self.on_connection_failed)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()

        # 버튼 활성화/비활성화 상태 변경
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)

    # 연결 해제
    def disconnect_from_backend(self):
        self.log_output.appendPlainText("연결을 끊는 중...")
        # 쓰레드가 존재하고 실행 중이면 정지
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread = None

        # 버튼 활성화/비활성화 상태 변경
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)

        # 디스플레이 영역을 초기화
        if hasattr(self, "video_display"):
            self.video_display.clear()
            self.video_display.setText("비디오 영역")

    # 연결 실패 시그널
    def on_connection_failed(self, msg):
        self.log_output.appendPlainText(msg)
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)

    # 쓰레드 종료 시그널
    def on_thread_finished(self):
        self.log_output.appendPlainText("연결이 끊어졌습니다.")
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)

        if self.thread:
            try: self.thread.change_pixmap_signal.disconnect(self.update_image)
            except TypeError: pass
            try: self.thread.connection_failed.disconnect(self.on_connection_failed)
            except TypeError: pass
            try: self.thread.finished.disconnect(self.on_thread_finished)
            except TypeError: pass
            self.thread = None

        if hasattr(self, "video_display"):
            self.video_display.clear()
            self.video_display.setText("비디오 영역")

    # 윈도우 닫기
    def closeEvent(self, event):
        print("윈도우를 닫습니다...")
        self.disconnect_from_backend()
        event.accept()

    # 이미지 업데이트
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        if hasattr(self, "video_display"):
            # 원본비율 유지 라벨에 맞춘 이미지 크기조절
            pixmap = qt_img.scaled(
                self.video_display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_display.setPixmap(pixmap)

    # cv -> qpixmap
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        # 한 줄의 바이트 수
        bytes_per_line = ch * w
        # np array -> qimage
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888
        )
        # qimage -> qpixmap
        return QtGui.QPixmap.fromImage(convert_to_Qt_format)

# 엔트리 포인트
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClientWindow()
    window.show()
    sys.exit(app.exec())