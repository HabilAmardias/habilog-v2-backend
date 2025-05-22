import torch
from abc import ABC, abstractmethod
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from PIL import Image
import pickle
from torchvision.models import mobilenet_v3_large
from torchvision.transforms import v2

class AbstractFAGUtil(ABC):
    @abstractmethod
    def preprocess_image(self, uploaded_image: BytesIO) -> torch.Tensor:
        pass
    @abstractmethod
    def get_features_extractor(self) -> torch.nn.Sequential:
        pass
    @abstractmethod
    def get_face_detector(self) -> LogisticRegression:
        pass
    @abstractmethod
    def get_age_classifier(self)-> torch.nn.Sequential:
        pass
    @abstractmethod
    def extract_features(self, feature_extractor: torch.nn.Sequential, tensor: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def check_if_face_exist(self, features: torch.Tensor, model: LogisticRegression) -> bool:
        pass
    @abstractmethod
    def classify_age(self, features: torch.Tensor, classifier: torch.nn.Sequential) -> tuple[float, str]:
        pass


class FAGUtilImpl(AbstractFAGUtil):
    def extract_features(self, feature_extractor, tensor):
        feature_extractor.eval()
        with torch.no_grad():
            pred: torch.Tensor = feature_extractor(tensor)
        return pred
    
    def check_if_face_exist(self, features, model):
        probs = model.predict(features.cpu().numpy())
        print(probs.item())
        return probs.item() >= 0.5
    
    def classify_age(self, features, classifier):
        age_map = {
            0: "0-20",
            1: "20-40",
            2: "40-60",
            3:"60+"
        }
        classifier.eval()
        with torch.no_grad():
            age_probs: torch.Tensor = classifier(features)

        wanted_index = age_probs.argmax().item()
        most_probable_age_range = age_map[wanted_index]
        prob = age_probs[wanted_index].item()

        return prob, most_probable_age_range
        
    def preprocess_image(self, uploaded_image):
        image = Image.open(uploaded_image).convert('RGB')
        
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor_image:torch.Tensor = transforms(image).unsqueeze(0)
        return tensor_image
    def get_features_extractor(self):
        model = mobilenet_v3_large()
        model.classifier[3] = torch.nn.Linear(1280, 4)
        
        model.load_state_dict(
            torch.load(
                'models/age_detector.pth',
                map_location="cpu",
                weights_only=True
            )
        )
        feature_extractor = torch.nn.Sequential()
        feature_extractor.add_module('features',model.features)
        feature_extractor.add_module('avgpool',model.avgpool)

        return feature_extractor
    def get_face_detector(self):
        f = open('models/classifier.pkl','rb')
        clf:LogisticRegression = pickle.load(f)
        f.close()
        return clf
    def get_age_classifier(self):
        model = mobilenet_v3_large()
        model.classifier[3] = torch.nn.Linear(1280, 4)
        
        model.load_state_dict(
            torch.load(
                'models/age_detector.pth',
                map_location="cpu",
                weights_only=True
            )
        )
        return model.classifier
    
def get_fag_util() -> AbstractFAGUtil:
    return FAGUtilImpl()