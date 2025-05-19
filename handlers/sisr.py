from abc import ABC, abstractmethod
from fastapi import UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from services.sisr import AbstractSISRService
from io import BytesIO
from errors.main import LargeImageResolutionError

class AbstractSISRHandler(ABC):
    @abstractmethod
    async def upscale_image(self, file: UploadFile) -> StreamingResponse:
        pass

class SISRHandlerImpl(AbstractSISRHandler):
    def __init__(self, sisr_service: AbstractSISRService):
        self.sisr_service = sisr_service
    
    async def upscale_image(self, file: UploadFile) -> StreamingResponse:
        if file.content_type not in ["image/png","image/jpeg","image/bmp"]:
            raise HTTPException(status_code=400)
        try:
            byte = BytesIO(await file.read())
            data = self.sisr_service.upscale_image(byte)
            return StreamingResponse(content=data,media_type=file.content_type)
        except LargeImageResolutionError as ler:
            raise HTTPException(status_code=400, detail=repr(ler))
        except Exception:
            raise HTTPException(status_code=500, detail="Internal Server Error")
        
        
    
def create_SISR_handler(service: AbstractSISRService) -> AbstractSISRHandler:
    return SISRHandlerImpl(service)
        