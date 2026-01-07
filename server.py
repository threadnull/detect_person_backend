
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

if __name__ == "__main__":
    import logging
    # rknnlite 라이브러리가 변경한 로깅 레벨 복원
    logging.addLevelName(logging.CRITICAL, "CRITICAL")
    logging.addLevelName(logging.ERROR, "ERROR")
    logging.addLevelName(logging.WARNING, "WARNING")
    logging.addLevelName(logging.INFO, "INFO")
    logging.addLevelName(logging.DEBUG, "DEBUG")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
