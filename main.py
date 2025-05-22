from routers.main import router
from utils.sisr import create_SISR_utils
from services.sisr import create_SISR_service
from handlers.sisr import create_SISR_handler
from fastapi import UploadFile, status
from fastapi.responses import JSONResponse
from utils.fag import get_fag_util
from services.fag import get_fag_service
from handlers.fag import get_fag_handler


sisr_util = create_SISR_utils()
sisr_service = create_SISR_service(sisr_util)
sisr_handler = create_SISR_handler(sisr_service)

fag_util = get_fag_util()
fag_service = get_fag_service(fag_util)
fag_handler = get_fag_handler(fag_service)

@router.post("/sisr/upload", status_code=status.HTTP_200_OK)
async def upscale_image(file: UploadFile):
    return await sisr_handler.upscale_image(file)

@router.post("/fag/upload", status_code=status.HTTP_200_OK)
async def classify_age_with_image(file: UploadFile):
    return await fag_handler.classify_age_with_image(file)

@router.get("/")
def root():
    return JSONResponse({
        "message":"hello world"
    })
    

