from abc import ABC, abstractmethod
from utils.sisr import AbstractSISRUtil
from io import BytesIO

class AbstractSISRService(ABC):
    @abstractmethod
    def upscale_image(self, upload: BytesIO) -> BytesIO:
        pass

class SISRServiceImpl(AbstractSISRService):
    def __init__(self, sisr_util: AbstractSISRUtil):
        self.sisr_util = sisr_util

    def upscale_image(self, upload: BytesIO):
        gen = self.sisr_util.load_model()
        preprocessed = self.sisr_util.preprocess_image(upload)
        out = self.sisr_util.upscale_image_with_generator(preprocessed, gen)
        return out
    
def create_SISR_service(util: AbstractSISRUtil) -> AbstractSISRService:
    return SISRServiceImpl(util)