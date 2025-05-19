from routers.main import router
from utils.sisr import create_SISR_utils
from services.sisr import create_SISR_service
from handlers.sisr import create_SISR_handler
from fastapi import UploadFile, status
from fastapi.responses import JSONResponse


sisr_util = create_SISR_utils()
sisr_service = create_SISR_service(sisr_util)
sisr_handler = create_SISR_handler(sisr_service)

@router.post("/sisr/upload", status_code=status.HTTP_200_OK)
async def upscale_image(file: UploadFile):
    return await sisr_handler.upscale_image(file)

@router.get("/")
def root():
    return JSONResponse({
        "message":"hello world"
    })
    

