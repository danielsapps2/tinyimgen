"""Dense face (478) and hand (21) landmarks with MediaPipe, in source-image pixels.

Models live in third_party/models (face_landmarker.task, hand_landmarker.task from
storage.googleapis.com/mediapipe-models). z is MediaPipe's relative depth, converted to pixels (negative = toward
the camera)."""
import os

import cv2
import numpy as np

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "models")
FACE_TASK = os.path.join(MODELS, "face_landmarker.task")
HAND_TASK = os.path.join(MODELS, "hand_landmarker.task")
_cache = {}


def _task(kind):
    if kind not in _cache:
        from mediapipe.tasks.python import BaseOptions, vision
        if kind == "face":
            _cache[kind] = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=FACE_TASK), num_faces=1, min_face_detection_confidence=0.3))
        else:
            _cache[kind] = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_TASK), num_hands=2, min_hand_detection_confidence=0.2,
                min_hand_presence_confidence=0.2))
    return _cache[kind]


def _run(kind, rgb):
    import mediapipe as mp
    return _task(kind).detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))


def face_mesh(rgb, box):
    """478 x 3 landmarks (x, y, z in source pixels) for the face in `box` (x, y, w, h, e.g. from YuNet), or None.
    The face is small in a full-body image, so it is cropped and upscaled first."""
    if not os.path.exists(FACE_TASK) or box is None:
        return None
    x, y, w, h = box[:4]
    S = max(w, h) * 2.2
    x0, y0 = x + w / 2 - S / 2, y + h / 2 - S / 2
    M = np.float32([[512 / S, 0, -x0 * 512 / S], [0, 512 / S, -y0 * 512 / S]])
    crop = cv2.warpAffine(rgb, M, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    r = _run("face", crop)
    if not r.face_landmarks:
        return None
    L = np.array([[p.x, p.y, p.z] for p in r.face_landmarks[0]], np.float64) * S
    L[:, 0] += x0
    L[:, 1] += y0
    return L


def face_triangles():
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
    e = C.FACE_LANDMARKS_TESSELATION
    return np.array([[e[i].start, e[i + 1].start, e[i + 2].start] for i in range(0, len(e), 3)], np.int64)


def face_oval():
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
    return np.array([c.start for c in C.FACE_LANDMARKS_FACE_OVAL], np.int64)


def hands(rgb):
    """List of 21 x 3 hand landmark arrays (x, y, z in source pixels). Each detection is refined on an upscaled
    crop, which places the fingers much better than the full-body pass."""
    if not os.path.exists(HAND_TASK):
        return []
    H, W = rgb.shape[:2]
    out = []
    for h in _run("hand", rgb).hand_landmarks:
        P = np.array([[p.x * W, p.y * H] for p in h])
        c, S = P.mean(0), max(np.ptp(P, 0).max() * 2.4, 64)
        x0, y0 = c - S / 2
        M = np.float32([[384 / S, 0, -x0 * 384 / S], [0, 384 / S, -y0 * 384 / S]])
        crop = cv2.warpAffine(rgb, M, (384, 384), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        r = _run("hand", crop).hand_landmarks
        best = None
        for h2 in r:  # the crop may also catch the other hand: keep the one nearest the first estimate
            Q = np.array([[p.x * S + x0, p.y * S + y0, p.z * S] for p in h2])
            d = np.linalg.norm(Q[:, :2].mean(0) - c)
            if best is None or d < best[0]:
                best = (d, Q)
        if best is not None and best[0] < S / 4:
            out.append(best[1])
        else:
            out.append(np.c_[P, [p.z * W for p in h]])
    return out


PARSE_MODEL = "jonathandinu/face-parsing"   # SegFormer, CelebAMask-HQ labels
NECK, CLOTH, HAIR = 17, 18, 13


def face_parse(rgb, box):
    """Per-pixel face-parsing labels (CelebAMask-HQ: 1 skin, 13 hair, 17 neck, 18 cloth, ...) around the face in
    `box`; 0 elsewhere. None if the model can't be loaded."""
    if box is None:
        return None
    import torch
    try:
        if "parse" not in _cache:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            _cache["parse"] = (SegformerImageProcessor.from_pretrained(PARSE_MODEL),
                               SegformerForSemanticSegmentation.from_pretrained(PARSE_MODEL).eval())
        proc, net = _cache["parse"]
    except Exception:
        return None
    from PIL import Image
    H, W = rgb.shape[:2]
    x, y, w, h = box[:4]
    S = max(w, h) * 2.4
    x0, y0 = max(0, int(x + w / 2 - S / 2)), max(0, int(y + h / 2 - S / 2))
    crop = rgb[y0:min(H, y0 + int(S)), x0:min(W, x0 + int(S))]
    with torch.no_grad():
        logits = net(**proc(images=Image.fromarray(crop), return_tensors="pt")).logits
    lab = torch.nn.functional.interpolate(logits, size=crop.shape[:2], mode="bilinear").argmax(1)[0].numpy()
    out = np.zeros((H, W), np.uint8)
    out[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = lab
    return out
