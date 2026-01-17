import os
import json
import cv2
import torch
from torchvision import models
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_tensor

def main():
    video_path = os.path.join("video_task", "data", "video.mp4")
    out_dir = os.path.join("video_task", "output")
    os.makedirs(out_dir, exist_ok=True)

    weights = models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    step = 10
    conf_th = 0.35

    results = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = to_tensor(frame_rgb)

        with torch.no_grad():
            pred = model([img])[0]

        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]

        keep = scores >= conf_th
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        frame_res = []
        for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            frame_res.append({"box": [float(x) for x in b], "score": float(s), "label_id": int(l)})

        results.append({"frame": frame_idx, "detections": frame_res})

        if len(results) <= 3:
            drawn = draw_bounding_boxes(
                (img * 255).byte(),
                boxes=boxes,
                labels=[f"id={int(l)}:{s:.2f}" for l, s in zip(labels, scores)],
                width=2
            )
            out_img = drawn.permute(1, 2, 0).cpu().numpy()
            out_path = os.path.join(out_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

        print(f"Processed frame {frame_idx}: {len(frame_res)} detections")
        frame_idx += 1

    cap.release()

    json_path = os.path.join(out_dir, "detections.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {json_path}")

if __name__ == "__main__":
    main()
