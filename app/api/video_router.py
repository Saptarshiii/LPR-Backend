from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.responses import Response
from services.video_service import process_video
import io

router = APIRouter()

@router.post("/process", summary="Return original uploaded video")
async def process_video_endpoint(file: UploadFile = File(...)):
    video_bytes = await process_video(file)
    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f"attachment; filename={file.filename}"
        }
    )