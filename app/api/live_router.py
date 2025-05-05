from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from services.live_service import generate_live_video

router = APIRouter()

@router.get("/live", summary="Live video stream with YOLO detection")
async def live_video_stream():
    return StreamingResponse(
        generate_live_video(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
