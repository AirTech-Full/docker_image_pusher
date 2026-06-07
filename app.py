from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import open_clip
import torch
import io
import base64

app = FastAPI()

# 启动时加载模型（权重已在 Dockerfile 预下载）
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model.eval()

DIMENSION = 512  # ViT-B/32 输出维度


class ImageRequest(BaseModel):
    image: str  # Base64 编码（不含 data:image/... 前缀）


class ImageBatchRequest(BaseModel):
    images: list[str]  # Base64 列表


@app.get("/health")
def health():
    return {"status": "ok", "model": "ViT-B-32", "dimension": DIMENSION}


@app.post("/embed/image")
def embed_image(req: ImageRequest):
    """单张图片 → 512 维向量"""
    embedding = _image_to_embedding(req.image)
    return {
        "dimension": DIMENSION,
        "vector": embedding
    }


@app.post("/embed/image_batch")
def embed_image_batch(req: ImageBatchRequest):
    """批量图片 → 512 维向量列表"""
    embeddings = [_image_to_embedding(img) for img in req.images]
    return {
        "dimension": DIMENSION,
        "count": len(embeddings),
        "vectors": embeddings
    }


def _image_to_embedding(image_b64: str) -> list[float]:
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
    embedding = embedding.squeeze().tolist()
    return embedding
