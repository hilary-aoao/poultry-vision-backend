import io
import time
from collections import defaultdict, deque
from typing import Dict, Optional

import torch
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import transforms

import chatbot_utils
from model_utils import CLASS_NAMES, load_model

app = FastAPI(title="PoultryVision API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = load_model()
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
LIMIT_PER_MIN = 20          # requests per person per minute
_hits = defaultdict(deque)


def rate_limit(request: Request):
    """Stop one visitor from using up the free Groq quota. Render sits behind a proxy, so use the forwarded address."""
    forwarded = request.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute and try again.")
    q.append(now)
    if len(_hits) > 5000:   # forget visitors we have not seen for a minute
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
            _hits.pop(k, None)


class ExplainRequest(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0, le=100)
    probabilities: Optional[Dict[str, float]] = None


class AskRequest(BaseModel):
    disease_class: str
    question: str = Field(min_length=3, max_length=300)


@app.get("/")
def read_root():
    return {"status": "PoultryVision backend is running"}


@app.post("/predict", dependencies=[Depends(rate_limit)])
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="That image is too large. The limit is 8 MB.")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="That file is not a readable image.")
    tensor = val_transforms(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(tensor), dim=1)[0]
    probabilities = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASS_NAMES))}
    predicted = CLASS_NAMES[int(torch.argmax(probs))]
    confidence = probabilities[predicted]
    info = chatbot_utils.confidence_tier(predicted, confidence)
    return {"predicted_class": predicted, "confidence": confidence, "tier": info["tier"],
            "reliability": info["reliability"], "probabilities": probabilities}


@app.post("/explain", dependencies=[Depends(rate_limit)])
def explain(req: ExplainRequest):
    if req.predicted_class not in CLASS_NAMES:
        raise HTTPException(status_code=422, detail=f"predicted_class must be one of {CLASS_NAMES}")
    return chatbot_utils.explain(req.predicted_class, req.confidence, req.probabilities)


@app.post("/ask", dependencies=[Depends(rate_limit)])
def ask(req: AskRequest):
    if req.disease_class not in CLASS_NAMES:
        raise HTTPException(status_code=422, detail=f"disease_class must be one of {CLASS_NAMES}")
    return chatbot_utils.ask(req.disease_class, req.question)
