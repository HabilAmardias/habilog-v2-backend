from entities.generator import Generator
import torch
from PIL import Image
from torchvision.transforms import v2
import numpy as np
from io import BytesIO
from abc import ABC, abstractmethod

class AbstractSISRUtil(ABC):
    @abstractmethod
    def upscale_image_with_generator(self, tensor: torch.Tensor, generator: Generator) -> BytesIO:
        pass
    @abstractmethod
    def load_model(self) -> Generator:
        pass
    @abstractmethod
    def preprocess_image(self, uploaded_file: BytesIO) -> torch.Tensor:
        pass

class SISRUtilImpl(AbstractSISRUtil):
    def load_model(self) -> Generator:
        generator = Generator(dim=64)
        generator.load_state_dict(
            torch.load('models/Generator.pth',map_location='cpu',weights_only=True)
        )
        return generator

    def preprocess_image(self, uploaded_file: BytesIO) -> torch.Tensor:
        image = Image.open(uploaded_file).convert('RGB')
        if image.height * image.width > 300*300:
            raise ValueError("Image resolution is too large")
        
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor_image:torch.Tensor = transforms(image).unsqueeze(0)
        return tensor_image

    def upscale_image_with_generator(self, tensor: torch.Tensor, generator: Generator) -> BytesIO:
        generator.eval()
        with torch.no_grad():
            pred: torch.Tensor = generator(tensor)
            pred = pred * 127.5 + 127.5
        pred_array = pred.permute(0,2,3,1).squeeze(0).cpu().numpy().astype(np.uint8)
        
        byte = BytesIO()
        out = Image.fromarray(pred_array)
        out.save(byte, format="PNG")
        byte.seek(0)

        return byte
    
def create_SISR_utils() -> AbstractSISRUtil:
    return SISRUtilImpl()
