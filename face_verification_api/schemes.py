from pydantic import BaseModel


class FaceVerificationResponse(BaseModel):
    """Face verification response schema"""

    verified: bool
    distance: float
    threshold: float
    model: str
    detector_backend: str
    similarity_metric: str
    time: float
    facial_areas: dict
