from fastapi import FastAPI
from api import image_router, video_router

app = FastAPI(title="Media Processor API")

app.include_router(image_router.router, prefix="/image", tags=["Image Processing"])
app.include_router(video_router.router, prefix="/video", tags=["Video Processing"])
