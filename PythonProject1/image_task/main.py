import torch
from torchvision import models
from PIL import Image
import json
import os

def main():
    img_path = os.path.join("image_task", "data", "image.jpg")

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    topk = torch.topk(probs, k=5)
    categories = weights.meta["categories"]

    result = []
    for p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        result.append({"label": categories[idx], "prob": float(p)})

    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

