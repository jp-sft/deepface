from dataclasses import dataclass, KW_ONLY, asdict


@dataclass
class DeepFaceResponse:
    _ = KW_ONLY
    verified: bool
    distance: float
    threshold: float
    model: str
    detector_backend: str
    similarity_metric: str
    time: float
    facial_areas: dict

    def to_dict(self):
        return asdict(self)
