from abc import ABC, abstractmethod
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from services.sisr import AbstractSISRService
from errors.main import WrongTypeError
from common.main import SuccessMessage
from base64 import b64encode

class AbstractSISRHandler(ABC):
    @abstractmethod
    def upscale_image(self, file: UploadFile) -> JSONResponse:
        pass

class SISRHandlerImpl(AbstractSISRHandler):
    def __init__(self, sisr_service: AbstractSISRService):
        self.sisr_service = sisr_service
    
    def upscale_image(self, file: UploadFile) -> JSONResponse:
        if file.content_type not in ["image/png","image/jpeg","image/bmp"]:
            return JSONResponse({
                "message": WrongTypeError,
                "data": None
            })
        data = self.sisr_service.upscale_image(file)
        return JSONResponse({
            "message": SuccessMessage,
            "data": b64encode(data.read()).decode("utf-8")
        })
    
def create_SISR_handler(service: AbstractSISRService) -> AbstractSISRHandler:
    return SISRHandlerImpl(service)
        