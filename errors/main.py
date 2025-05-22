class LargeImageResolutionError(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class NoFaceDetected(Exception):
    def __init__(self, *args):
        super().__init__(*args)