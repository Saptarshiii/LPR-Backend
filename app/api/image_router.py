from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from services.image_service import process_image
import io

router = APIRouter()

@router.post("/process", summary="Process an image and return result")
async def process_image_endpoint(file: UploadFile = File(...)):
    image_bytes = await process_image(file)
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
