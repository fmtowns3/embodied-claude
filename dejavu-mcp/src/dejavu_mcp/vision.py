"""モデル層：画像から scene / person / face の3ベクトルを作る。

実験スクリプト（embed_faces / patch_search / cluster …）に散っていた処理を
1か所に集めた。dejavu-mcp の中核になる部分。

3ベクトルの作り方は heishio 版と1点だけ変えてある。
彼は「人物以外をニュートラルグレーで塗った画像」を丸ごと embed して
全256パッチの平均を取るが、それだとグレーのパッチが平均に混ざって薄まる。
ここでは塗らずに、人物マスクに入るパッチだけを選んで平均する。
（塗り潰しが max 検索に効かないことは 2026-08-18 に実測済み。平均には効く。）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

# MCP サーバーは任意の作業ディレクトリから起動されるので、パスを相対で決め打ちできない。
# 既定はこのコンポーネント直下（dejavu-mcp/models/）だが、環境変数で差し替えられる。
MODEL_DIR = Path(os.environ.get("DEJAVU_MODEL_DIR",
                                Path(__file__).resolve().parents[2] / "models"))
FACE_MODEL = MODEL_DIR / "blaze_face_short_range.tflite"
SEG_MODEL = MODEL_DIR / "selfie_segmenter.tflite"

DINO_NAME = "facebook/dinov2-with-registers-base"
DINO_DIM = 768
NUM_REGISTERS = 4
GRID = 16
CROP_MARGIN = 0.35

_TO_TENSOR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


@dataclass
class Observation:
    """1枚の画像から取れたもの。ベクトルは無い場合がある（顔が写っていない等）。"""
    capture_path: str
    person_ratio: float
    face_confidence: float
    vectors: dict[str, np.ndarray]  # kind -> vector


class Vision:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(DINO_NAME).to(self.device).eval()
        self._face = mp.tasks.vision.FaceDetector.create_from_options(
            mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_detection_confidence=0.5))
        self._seg = mp.tasks.vision.ImageSegmenter.create_from_options(
            mp.tasks.vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(SEG_MODEL)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                output_confidence_masks=True))

    def close(self) -> None:
        self._face.close()
        self._seg.close()

    def __enter__(self) -> "Vision":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- 低レベル ---

    def patches(self, image: Image.Image) -> np.ndarray:
        """画像 → L2正規化済みパッチ特徴 [256, 768]"""
        tensor = _TO_TENSOR(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(pixel_values=tensor)
        p = out.last_hidden_state[0, 1 + NUM_REGISTERS:].cpu().numpy()
        return p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-10, None)

    def person_cover(self, image: Image.Image) -> tuple[np.ndarray, float]:
        """人物マスクをパッチ格子(16x16)に落とす。(被覆率ベクトル, 画素ベースの人物率)"""
        rgb = np.asarray(image)
        result = self._seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        mask = result.confidence_masks[0].numpy_view()
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        binary = (mask > 0.5).astype(np.float32)
        cover = cv2.resize(binary, (GRID, GRID), interpolation=cv2.INTER_AREA).reshape(-1)
        return cover, float(binary.mean())

    def face_crop(self, image: Image.Image) -> tuple[Image.Image | None, float]:
        rgb = np.asarray(image)
        result = self._face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.detections:
            return None, 0.0
        best = max(result.detections, key=lambda d: d.categories[0].score)
        b = best.bounding_box
        h, w = rgb.shape[:2]
        mx, my = int(b.width * CROP_MARGIN), int(b.height * CROP_MARGIN)
        crop = rgb[max(0, b.origin_y - my):min(h, b.origin_y + b.height + my),
                   max(0, b.origin_x - mx):min(w, b.origin_x + b.width + mx)]
        if crop.size == 0:
            return None, 0.0
        return Image.fromarray(crop), float(best.categories[0].score)

    # --- 高レベル ---

    def face_in_person(self, image: Image.Image, cover: np.ndarray) -> float:
        """検出された顔の枠が、人物マスクにどれだけ重なっているか（0〜1）。

        ★顔検出は誤検出する。実測例：車内の写真で「車の天井とサンバイザー」を
        顔として検出し、そのスコア 0.685 は同じ人の本物の顔 0.625 より高かった。
        スコアでは弾けない。しかし顔は人物領域の中にあるはずなので、
        既に計算しているマスクで検証できる。
        """
        rgb = np.asarray(image)
        result = self._face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.detections:
            return 0.0
        b = max(result.detections, key=lambda d: d.categories[0].score).bounding_box
        h, w = rgb.shape[:2]
        grid = cover.reshape(GRID, GRID)
        c0, c1 = int(b.origin_x / w * GRID), int(np.ceil((b.origin_x + b.width) / w * GRID))
        r0, r1 = int(b.origin_y / h * GRID), int(np.ceil((b.origin_y + b.height) / h * GRID))
        cell = grid[max(0, r0):min(GRID, r1), max(0, c0):min(GRID, c1)]
        return float(cell.mean()) if cell.size else 0.0

    def observe(self, path: Path, face_in_person_min: float = 0.25) -> Observation:
        """1枚から scene / person / face を作る。取れないものは入れない。

        face_in_person_min: 顔の枠が人物マスクにこれ未満しか重なっていなければ、
        誤検出とみなして face を捨てる。
        """
        image = Image.open(path).convert("RGB")
        patches = self.patches(image)
        cover, ratio = self.person_cover(image)

        vectors: dict[str, np.ndarray] = {}
        inside = patches[cover > 0.5]
        outside = patches[cover <= 0.5]
        if len(inside):
            vectors["person"] = unit(inside.mean(axis=0))
        if len(outside):
            vectors["scene"] = unit(outside.mean(axis=0))

        crop, conf = self.face_crop(image)
        if crop is not None:
            overlap = self.face_in_person(image, cover)
            if overlap >= face_in_person_min:
                vectors["face"] = unit(self.patches(crop).mean(axis=0))
            else:
                conf = 0.0  # 誤検出として捨てた印

        return Observation(str(path), ratio, conf, vectors)
