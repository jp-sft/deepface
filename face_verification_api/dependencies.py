import enum
from dataclasses import dataclass, KW_ONLY, asdict
from typing import Annotated

from fastapi import Depends

from .service import DeepFaceVerifyService


class _ModelName(str, enum.Enum):
    VGG_FACE = 'VGG-Face'
    FACENET = "Facenet"
    FACENET_512 = "Facenet512"
    OPEN_FACE = "OpenFace"
    DEEP_FACE = "DeepFace"
    DEEP_ID = "DeepID"
    DLIB = "Dlib"
    ARC_FACE = "ArcFace"
    S_FACE = "SFace"


class _DetectorBackend(str, enum.Enum):
    OPENCV = "opencv"
    RETINAFACE = "retinaface"
    MTCNN = "mtcnn"
    SSD = "ssd"
    DLIB = "dlib"
    MEDIAPIPE = "mediapipe"
    YOLOV_8 = "yolov8"


class _DistanceMetric(str, enum.Enum):
    COSINE = "cosine"
    euclidean = "euclidean"
    euclidean_l2 = "euclidean_l2"


@dataclass
class _DeepFaceCommon:
    _ = KW_ONLY
    model_name: _ModelName = _ModelName.VGG_FACE
    detector_backend: _DetectorBackend = _DetectorBackend.OPENCV
    distance_metric: _DistanceMetric = _DistanceMetric.COSINE
    align: bool = True
    enforce_detection: bool = False


_DeepFaceCommonDepends = Annotated[_DeepFaceCommon, Depends(_DeepFaceCommon)]


def _get_service(q: _DeepFaceCommonDepends):
    data = asdict(q)
    return DeepFaceVerifyService(**data)


ServiceDepends = Annotated[DeepFaceVerifyService, Depends(_get_service)]
