"""
FastAPI app for face verification

API endpoints:
    - /verify
      POST:
        Request body: JSON
        payload: {
            "image_file": "image",
            "known_image_files": ["image1", "image2", ...],
          }
"""
import asyncio
import logging.config
import os

# import importlib.metadata
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .domain import DeepFaceResponse
from .dependencies import ServiceDepends
from .schemes import FaceVerificationResponse

# _DISTRIBUTION_METADATA = importlib.metadata.metadata("deepface")
#
# __version__ = _DISTRIBUTION_METADATA['version']
# __title__ = _DISTRIBUTION_METADATA['name']
# __summary__ = _DISTRIBUTION_METADATA['summary']
# __description__ = _DISTRIBUTION_METADATA['description']

_logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(
    # title=__title__,
    # summary=__summary__,
    # description=__description__,
    # version=__version__,

    docs_url="/",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/verify", response_model=FaceVerificationResponse, status_code=status.HTTP_200_OK)
async def _verify(
        service: ServiceDepends,
        image_file: UploadFile = File(...),
        known_image_files: list[UploadFile] = File(...),
) -> FaceVerificationResponse:
    """Verify if the given image is of the given person
    """
    # Load the images
    image_content = await image_file.read()
    known_image_contents: tuple[bytes] = await asyncio.gather(
        *[known_image_file.read() for known_image_file in known_image_files]
    )
    try:
        response = service.verify(image_content, known_image_contents)
    except Exception as e:
        error_bad_thing_happened = "Error: Bad thing happened"
        _logger.exception(error_bad_thing_happened)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_bad_thing_happened)
    return FaceVerificationResponse(**response.to_dict())