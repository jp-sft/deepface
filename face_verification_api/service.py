from dataclasses import dataclass, KW_ONLY, field
from io import BytesIO

import numpy as np
from PIL import Image

from deepface import DeepFace
from .domain import DeepFaceResponse


@dataclass
class DeepFaceVerifyService:
    _ = KW_ONLY
    model_name: str
    detector_backend: str
    distance_metric: str
    align: bool
    enforce_detection: bool = field(default=False)

    @staticmethod
    def _load_image_into_numpy_array(data):
        return np.array(Image.open(BytesIO(data)))

    def verify(self, image_content, known_image_contents):

        match_result = dict()

        image = self._load_image_into_numpy_array(image_content)
        for known_image_content in known_image_contents:
            known_image = self._load_image_into_numpy_array(known_image_content)
            res = DeepFace.verify(
                img1_path=image,
                img2_path=known_image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                align=self.align,
                enforce_detection=False
            )
            if res['verified']:
                match_result = res
                break
            if not match_result or match_result['distance'] > res['distance']:
                match_result = res
        return DeepFaceResponse(**match_result)
