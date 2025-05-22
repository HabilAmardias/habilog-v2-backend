from abc import ABC, abstractmethod
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from errors.main import NoFaceDetected
from services.fag import AbstractFAGService

class AbstractFAGHandler(ABC):
    @abstractmethod
    async def classify_age_with_image(self, file: UploadFile):
        pass

class FAGHandlerImpl(AbstractFAGHandler):
    def __init__(self, service: AbstractFAGService):
        self.service = service
    async def classify_age_with_image(self, file):
        if file.content_type not in ["image/png","image/jpeg","image/bmp"]:
            raise HTTPException(status_code=400, detail="This file is not an image file")
        try:
            byte = BytesIO(await file.read())
            prob, age_range = self.service.classify_age_with_image(byte)
            return JSONResponse({"message":"success","data":{"probability": prob, "age_range": age_range}})
        except NoFaceDetected:
            raise HTTPException(400, "No Face Detected")
        except Exception:
            raise HTTPException(status_code=500, detail="Internal Server Error")

def get_fag_handler(service: AbstractFAGService) -> AbstractFAGHandler:
    return FAGHandlerImpl(service)