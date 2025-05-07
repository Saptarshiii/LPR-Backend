from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from services.image_service import process_image_with_ocr
import io

router = APIRouter()

@router.post("/process", summary="Process an image and return result")
async def process_image_endpoint(file: UploadFile = File(...)):
    #image_bytes = await process_image_with_ocr(file)
    #image_bytes = await process_image_with_ocr(await file.read())
    image_bytes = await file.read()
    result = process_image_with_ocr(image_bytes)


    return StreamingResponse(io.BytesIO(result), media_type="image/png")
