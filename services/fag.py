from abc import ABC, abstractmethod
from io import BytesIO
from utils.fag import AbstractFAGUtil
from errors.main import NoFaceDetected

class AbstractFAGService(ABC):
    @abstractmethod
    def classify_age_with_image(self, uploaded_image: BytesIO) -> tuple[float, str]:
        pass

class FAGServiceImpl(AbstractFAGService):
    def __init__(self, util: AbstractFAGUtil):
        self.util = util
    
    def classify_age_with_image(self, uploaded_image):
        image_tensor = self.util.preprocess_image(uploaded_image)
        extractor = self.util.get_features_extractor()
        classifier = self.util.get_age_classifier()

        features = self.util.extract_features(extractor, image_tensor)
        is_face_exist = self.util.check_if_face_exist(features)
        if not is_face_exist:
            raise NoFaceDetected("No Face Detected From Image")
        prob, age_range = self.util.classify_age(features, classifier)
        return prob, age_range
    
def get_fag_service(util: AbstractFAGUtil) -> AbstractFAGService:
    return FAGServiceImpl(util)
        