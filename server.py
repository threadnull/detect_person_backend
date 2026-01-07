import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
from inference import Detector

detector: Optional[Detector] = None

# 리소스 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    print("Application startup...")
    try:
        detector = Detector()
        print("Detector initialized")
    except Exception as e:
        print(f"Failed to initialize Detector: {e}")
        detector = None
    
    yield
    
    print("Application shutdown...")
    if detector:
        detector.release()

app = FastAPI(lifespan=lifespan)

# 파비콘 사용안함
@app.get('/favicon.ico', include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/")
def video_feed():
    """
    비디오 스트리밍 엔드포인트
    video_generator를 호출
    multipart/x-mixed-replace 형식의 스트리밍 응답 반환
    """
    if detector is None:
        return {"error": "Detector not initialized. Please check server logs."}
        
    return StreamingResponse(
        detector.generate_video_frame(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/log")
def get_log():
    """
    로그 스트리밍 엔드포인트
    Detector의 큐를 확인하여 텍스트 데이터를 클라이언트에게 전송
    """
    if detector is None:
        return Response("Detector not initialized", status_code=500)
    
    def iter_log():
        while True:
            # 큐가 비어있지 않으면 메시지 꺼내기
            if not detector.log_queue.empty*():
                msg = detector.log_queue.get()
                # 텍스트 라인 단위
                yield f"data: {msg}\n"

            else:
                # cpu 과점유 방지
                time.sleep(0.1)

    return StreamingResponse(
        iter_log(),
        media_type="text/plain; charset=utf-8"
    )

if __name__ == "__main__":
    import logging
    # rknnlite 라이브러리가 변경한 로깅 레벨 복원
    logging.addLevelName(logging.CRITICAL, "CRITICAL")
    logging.addLevelName(logging.ERROR, "ERROR")
    logging.addLevelName(logging.WARNING, "WARNING")
    logging.addLevelName(logging.INFO, "INFO")
    logging.addLevelName(logging.DEBUG, "DEBUG")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
