from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import ORJSONResponse
import cv2
import numpy as np
import base64
from src.core.enhancer import process_image_pipeline
from src.api.schemas import EnhancerResponse

router = APIRouter()

@router.post("/process", response_model=EnhancerResponse)
async def process_image(
    file: UploadFile = File(...),
    algorithm: str = Form("auto"),
    strength: float = Form(1.0)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Invalid image format.")

    payload = process_image_pipeline(img_bgr, algorithm=algorithm, strength=float(strength))

    _, encoded_img = cv2.imencode(".jpg", payload.image_data)
    img_b64 = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
    
    from dataclasses import asdict
    return EnhancerResponse(
        image_base64=f"data:image/jpeg;base64,{img_b64}",
        metrics=asdict(payload.metrics),
        histograms=asdict(payload.histograms)
    )
