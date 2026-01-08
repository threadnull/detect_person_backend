import os
import traceback
import cv2
import numpy as np
import queue
import datetime
from rknnlite.api import RKNNLite

# 하이퍼 파라미터
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 480
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45
MODEL_PATH = "./model/yolo11n_rk3588.rknn"
CALIB_PATH = "./calib/stereo_calib.npz"
COLOR_BOX = (0, 255, 0)
COLOR_TEXT = (0, 255, 0)
COLOR_MASK = (0, 0, 255)
LOG_INTERVAL = 2

MASK_POINTS = [
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
]

# 전처리
def _post_process(outputs, conf_threshold, nms_threshold):
    if outputs is None or len(outputs) == 0:
        return [], [], []
    
    try:
        predictions = np.squeeze(outputs[0]).T
        if predictions.shape[0] == 0:
            return [], [], []
        
        scores_raw = predictions[:, 4:]
        max_scores = np.max(scores_raw, axis=1)
        class_ids = np.argmax(scores_raw, axis=1)

        mask = (max_scores > conf_threshold) & (class_ids == 0)

        preds = predictions[mask]
        if len(preds) == 0:
            return [], [], []
        
        w = preds[:, 2]
        h = preds[:, 3]
        x = preds[:, 0] - w/2
        y = preds[:, 1] - h/2

        boxes = np.stack((x, y, w, h), axis=1).tolist()
        confidences = max_scores[mask].tolist()

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        person_boxes = [boxes[i] for i in indices]
        return person_boxes, [], []
    
    except Exception as e:
        print(f"post_process Error: {e}")
        traceback.print_exc()
        return [], [], []

# 감지 + 거리 + 마스킹 클레스
class Detector:
    def __init__(self):
        self.rknn_lite = RKNNLite()
        self.cap = None
        self.log_queue = queue.Queue()

        # 로그 관련 변수 초기화 (util.py 로직)
        self.last_log_time = None
        self.log_deduplication_interval = datetime.timedelta(seconds=LOG_INTERVAL)
        
        self._load_model()
        self._load_calibration()
        self._init_camera()
        self._init_stereo()
        self.mask_points_np = np.array(MASK_POINTS, np.int32).reshape((-1, 1, 2))

    # 모델 로드
    def _load_model(self):
        if self.rknn_lite.load_rknn(MODEL_PATH) != 0:
            raise RuntimeError("Error: Model load fail")
        if self.rknn_lite.init_runtime() != 0:
            raise RuntimeError("Error: Init runtime fail")
        print("Model loaded successfully.")

    # 캘리 데이터 로드
    def _load_calibration(self):
        if not os.path.exists(CALIB_PATH):
            raise FileNotFoundError("Error: Calibration file not found")
        
        data = np.load(CALIB_PATH)
        self.mtxL, self.distL = data['mtxL'], data['distL']
        self.mtxR, self.distR = data['mtxR'], data['distR']
        self.R, self.T = data['R'], data['T']
        
        self.focal_length = self.mtxL[0, 0]
        self.baseline = abs(self.T[0].item())
        print("Calibration data loaded successfully.")

    # 카메라 초기화
    def _init_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Camera open failed")
        
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.raw_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.raw_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.half_width = self.raw_width // 2

        print(f"Resolution {self.half_width}x{self.raw_height}")

        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            self.mtxL, self.distL, self.mtxR, self.distR,
            (self.half_width, self.raw_height), self.R, self.T, alpha=0
        )
        self.map1_L, self.map2_L = cv2.initUndistortRectifyMap(
            self.mtxL, self.distL, R1, P1, (self.half_width, self.raw_height), cv2.CV_16SC2
        )
        self.map1_R, self.map2_R = cv2.initUndistortRectifyMap(
            self.mtxR, self.distR, R2, P2, (self.half_width, self.raw_height), cv2.CV_16SC2
        )

    # SGBM 초기화
    def _init_stereo(self):
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 3,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2
        )

    # 자원 반환
    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("Camera released.")
        if self.rknn_lite:
            self.rknn_lite.release()
            print("Model released.")

    # 비디오 프레임 생성
    def generate_video_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            try:
                imgL_rect, processed_boxes = self._process_frame(frame)

                log_distances = [item['dist'] for item in processed_boxes if item.get('dist') and item['dist'] != "Calc..."]
                if log_distances:
                    current_time = datetime.datetime.now()
                    
                    # 마지막 로그 시간과 비교하여 일정 시간 지났는지 확인
                    if self.last_log_time is None or (current_time - self.last_log_time) > self.log_deduplication_interval:
                        time_str = current_time.strftime("%H:%M:%S")
                        dist_msg = ", ".join(log_distances)
                        full_msg = f"[{time_str}] Person Detected: {dist_msg}"

                        # 큐에 로그 저장 (서버가 가져갈 수 있도록)
                        if self.log_queue.qsize() < 10: # 큐가 너무 쌓이지 않게 관리
                            self.log_queue.put(full_msg)
            
                        self.last_log_time = current_time

                self._draw_results(imgL_rect, processed_boxes, self.half_width, self.raw_height)
                _, jpeg = cv2.imencode('.jpg', imgL_rect)

                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
            except Exception as e:
                print(f"Error in video generator loop: {e}")
                traceback.print_exc()
                break

    # 프레임 처리
    def _process_frame(self, frame):
        imgL_raw = frame[:, :self.half_width]
        imgR_raw = frame[:, self.half_width:]

        # 카메라 보정
        imgL_rect = cv2.remap(imgL_raw, self.map1_L, self.map2_L, cv2.INTER_LINEAR)
        imgR_rect = cv2.remap(imgR_raw, self.map1_R, self.map2_R, cv2.INTER_LINEAR)

        # 오버레이 마스킹
        overlay = imgL_rect.copy()
        cv2.fillPoly(overlay, [self.mask_points_np], COLOR_MASK)
        cv2.addWeighted(overlay, 0.3, imgL_rect, 0.7, 0, imgL_rect)

        # 추론
        img_rgb = cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_rgb, INPUT_SIZE)
        img_input = np.expand_dims(img_input, axis=0)

        outputs = self.rknn_lite.inference(inputs=[img_input])
        boxes, _, _ = _post_process(outputs, CONF_THRESHOLD, NMS_THRESHOLD)
        
        processed_boxes = []
        if boxes:
            disparity = self._calculate_disparity(imgL_rect, imgR_rect, self.half_width, self.raw_height)
            for box in boxes:
                dist_str = self._calculate_distance(box, disparity, self.half_width, self.raw_height)
                processed_boxes.append({'box': box, 'dist': dist_str})
        
        return imgL_rect, processed_boxes

    # 거리 계산
    def _calculate_disparity(self, imgL, imgR, width, height):
        scale_percent = 0.5
        # 이미지 크기 절반
        w_small = int(width * scale_percent)
        h_small = int(height * scale_percent)
        imgL_small = cv2.resize(imgL, (w_small, h_small), interpolation=cv2.INTER_AREA)
        imgR_small = cv2.resize(imgR, (w_small, h_small), interpolation=cv2.INTER_AREA)
        
        return self.stereo.compute(imgL_small, imgR_small)

    def _calculate_distance(self, box, disparity, width, height):
        scale_x = width / INPUT_SIZE[0]
        scale_y = height / INPUT_SIZE[1]
        
        x, y, w, h = box
        x1, y1 = int(x * scale_x), int(y * scale_y)
        x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        if cv2.pointPolygonTest(self.mask_points_np, (center_x, center_y), False) >= 0:
            return None

        scale_percent = 0.5
        disp_scale_x = (width * scale_percent) / width
        
        box_w, box_h = x2 - x1, y2 - y1
        crop_w, crop_h = int(box_w * 0.5), int(box_h * 0.5)
        d_x1 = int((center_x - crop_w // 2) * disp_scale_x)
        d_y1 = int((center_y - crop_h // 2) * disp_scale_x)
        d_x2 = int((center_x + crop_w // 2) * disp_scale_x)
        d_y2 = int((center_y + crop_h // 2) * disp_scale_x)
        
        d_x1, d_y1 = max(0, d_x1), max(0, d_y1)
        d_x2 = min(disparity.shape[1], d_x2)
        d_y2 = min(disparity.shape[0], d_y2)

        if d_x2 > d_x1 and d_y2 > d_y1:
            roi = disparity[d_y1:d_y2, d_x1:d_x2]
            valid_disp = roi[roi > 0]
            if len(valid_disp) > 0:
                median_disp = np.median(valid_disp) / 16.0
                f_small = self.focal_length * scale_percent
                if median_disp > 0:
                    dist_m = (f_small * self.baseline) / (median_disp * 1000.0)
                    return f"{dist_m:.2f}m"
        return "Calc..."

    # 결과 출력
    def _draw_results(self, image, processed_boxes, width, height):
        scale_x = width / INPUT_SIZE[0]
        scale_y = height / INPUT_SIZE[1]

        for item in processed_boxes:
            dist_str = item.get('dist')
            if dist_str is None:
                continue

            x, y, w, h = item['box']
            x1, y1 = int(x * scale_x), int(y * scale_y)
            x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_BOX, 2)
            cv2.putText(image, dist_str, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
            cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)

