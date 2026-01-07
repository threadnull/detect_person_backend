import os
import datetime
import cv2
import numpy as np
from rknnlite.api import RKNNLite
from PyQt6.QtCore import QThread, pyqtSignal

# 전처리
def post_process(outputs, conf_threshold, nms_threshold):
    predictions = np.squeeze(outputs[0]).T
    scores_raw = predictions[:, 4:]
    max_scores = np.max(scores_raw, axis=1)
    class_ids = np.argmax(scores_raw, axis=1)

    # 사람만 필터링
    mask = (max_scores > conf_threshold) & (class_ids == 0)

    preds = predictions[mask]
    scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(preds) == 0:
        return [], [], []
    
    w = preds[:, 2]
    h = preds[:, 3]
    x = preds[:, 0] - w/2
    y = preds[:, 1] - h/2

    boxes = np.stack((x, y, w, h), axis=1).tolist()
    confidences = scores.tolist()

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    person_boxs = []
    person_scores = []
    person_cls_ids = []

    if len(indices) > 0:
        for i in indices.ravel():
            person_boxs.append(boxes[i])
            person_scores.append(confidences[i])
            person_cls_ids.append(class_ids[i])

    return person_boxs, person_scores, person_cls_ids

# 추론 + 깊이
class DetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)

    def __init__(
            self, log_interval,
            model_path,
            calib_path,
            camera_index=0,
            frame_width=1280,
            frame_height=480,
            input_size=(640, 640),
            conf_threshold=0.35,
            nms_threshold=0.45,
            color_box=(0, 255, 0),
            color_text=(0, 255, 0)
        ):
        super().__init__()
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.model_path = model_path
        self.calib_path = calib_path
        self.color_box = color_box
        self.color_text = color_text
        self._run_flag = True
        self.last_log_time = None
        self.log_deduplication_interval = datetime.timedelta(seconds=log_interval)
        
        # SGBM init
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 3,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def run(self):
        rknn_lite = RKNNLite()
        
        # 모델 로드
        if rknn_lite.load_rknn(self.model_path) != 0:
            self.log_signal.emit("Error: Model load fail")
            return
        if rknn_lite.init_runtime() != 0:
            self.log_signal.emit("Error: Init runtime fail")
            return

        # 캘리 데이터 로드
        if not os.path.exists(self.calib_path):
             self.log_signal.emit("Error: Calib file not found")
             return

        data = np.load(self.calib_path)
        mtxL, distL = data['mtxL'], data['distL']
        mtxR, distR = data['mtxR'], data['distR']
        R, T = data['R'], data['T']

        # 초점거리, 베이스라인 추출
        focal_length = mtxL[0, 0]
        baseline = abs(T[0])

        # 카메라 init
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if raw_width == 0:
            self.log_signal.emit("Error: Camera open failed")
            return

        half_width = raw_width // 2
        self.log_signal.emit(f"Resolusion {half_width}x{raw_height}")
        self.log_signal.emit(f"")

        # Rectification 변환 행렬 계산
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtxL, distL, mtxR, distR, (half_width, raw_height), R, T, alpha=0
        )
        map1_L, map2_L = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (half_width, raw_height), cv2.CV_16SC2)
        map1_R, map2_R = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (half_width, raw_height), cv2.CV_16SC2)

        while self._run_flag and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 좌우 분할
            imgL_raw = frame[:, :half_width]
            imgR_raw = frame[:, half_width:]

            # 카메라 보정
            imgL_rect = cv2.remap(imgL_raw, map1_L, map2_L, cv2.INTER_LINEAR)
            imgR_rect = cv2.remap(imgR_raw, map1_R, map2_R, cv2.INTER_LINEAR)

            # 추론
            img_rgb = cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2RGB)
            img_input = cv2.resize(img_rgb, self.input_size)
            img_input = np.expand_dims(img_input, axis=0)

            outputs = rknn_lite.inference(inputs=[img_input])
            boxes, scores, class_ids = post_process(outputs, self.conf_threshold, self.nms_threshold)

            # 거리 계산
            scale_percent = 0.5
            w_small = int(half_width * scale_percent)
            h_small = int(raw_height * scale_percent)
            
            imgL_small = cv2.resize(imgL_rect, (w_small, h_small), interpolation=cv2.INTER_AREA)
            imgR_small = cv2.resize(imgR_rect, (w_small, h_small), interpolation=cv2.INTER_AREA)
            
            disparity = self.stereo.compute(imgL_small, imgR_small)
            
            # 스케일 팩터
            scale_x = half_width / self.input_size[0]
            scale_y = raw_height / self.input_size[1]
            disp_scale_x = w_small / half_width
            disp_scale_y = h_small / raw_height

            # 거리 로그
            log_distances = []

            for box in boxes:
                x, y, w, h = box
                
                # 좌표 복원
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + w) * scale_x)
                y2 = int((y + h) * scale_y)
                
                # 박스 roi
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                box_w, box_h = x2 - x1, y2 - y1
                crop_w, crop_h = int(box_w * 0.5), int(box_h * 0.5)

                d_x1 = int((center_x - crop_w // 2) * disp_scale_x)
                d_y1 = int((center_y - crop_h // 2) * disp_scale_y)
                d_x2 = int((center_x + crop_w // 2) * disp_scale_x)
                d_y2 = int((center_y + crop_h // 2) * disp_scale_y)

                # 예외 처리
                d_x1 = max(0, d_x1); d_y1 = max(0, d_y1)
                d_x2 = min(w_small, d_x2); d_y2 = min(h_small, d_y2)

                text_str = "Calc..."

                if d_x2 > d_x1 and d_y2 > d_y1:
                    roi = disparity[d_y1:d_y2, d_x1:d_x2]
                    valid_disp = roi[roi > 0]
                    
                    if len(valid_disp) > 0:
                        median_disp = np.median(valid_disp) / 16.0
                        f_small = focal_length * scale_percent
                        
                        if median_disp > 0:
                            dist_mm = (f_small * baseline) / median_disp
                            # numpy -> float
                            dist_m = float(dist_mm / 1000.0)
                            
                            text_str = f"{dist_m:.2f}m"
                            log_distances.append(f"{dist_m:.1f}m")

                # 단순 거리 표시
                cv2.rectangle(imgL_rect, (x1, y1), (x2, y2), self.color_box, 2)
                cv2.putText(imgL_rect, text_str, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_text, 2)

            # 로그 기록
            if log_distances:
                current_time = datetime.datetime.now()
                if self.last_log_time is None or (current_time - self.last_log_time) > self.log_deduplication_interval:
                    time_str = current_time.strftime("%H:%M:%S")
                    dist_msg = ", ".join(log_distances)
                    full_msg = f"[{time_str}] Detected: {dist_msg}"
                    self.log_signal.emit(full_msg)
                    self.last_log_time = current_time

            self.change_pixmap_signal.emit(imgL_rect)

        cap.release()
        rknn_lite.release()

    def stop(self):
        self._run_flag = False
        self.wait()