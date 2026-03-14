#!/usr/bin/env python3
"""
Runpod GR00T WebSocket Server for R1 + `Beable/dexterous_ee_56`

Bu script Runpod makinesinde çalışır:
- Lokal makineden gelen 17 boyutlu `observation.state` + RGB görüntüyü alır,
- GR00T'in Eagle + GrootPackInputs pipeline'ını taklit ederek batch hazırlar,
- GR00T policy (`policy.type=groot`) ile aksiyon üretir,
- Aksiyonu dataset stats ile inverse MIN_MAX unnormalize ederek 17 boyutlu liste olarak geri yollar.

Kullanım (Runpod içinde, `groot` env açıkken):

    python runpod_groot_server.py

NOT:
- `PRETRAINED_POLICY_PATH` değerini kendi GR00T checkpoint yoluna göre güncelle.
- Dataset stats, Hugging Face'ten `Beable/dexterous_ee_56` dataset'inin
  `meta/stats.json` dosyasından otomatik çekilir (revision=3.1).
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import websockets
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.constants import HF_LEROBOT_HOME


# --------------------------------------------------------------------------------------
# CONFIG – Runpod ortamına göre ÖNEMLİ AYARLAR
# --------------------------------------------------------------------------------------

# GR00T fine‑tune checkpoint yolu (Runpod içindeki path)
# Örnek (eğitimden sonra):
#   /workspace/Gr00t-train/outputs/groot_beable/checkpoints/040000/pretrained_model
PRETRAINED_POLICY_PATH = Path(
    "/workspace/Gr00t-train/outputs/groot_beable/checkpoints/040000/pretrained_model"
)

# Hugging Face dataset – GR00T'i bununla eğittik
DATASET_REPO_ID = "Beable/dexterous_ee_56"
DATASET_REVISION = "v3.1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GrootPackInputsStep ile uyumlu sabitler
MAX_STATE_DIM = 64
MAX_ACTION_DIM = 32
STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
EMBODIMENT_TAG = "new_embodiment"
EMBODIMENT_ID = 31  # GrootPackInputsStep.embodiment_mapping["new_embodiment"]

# Global policy & eagle processor & stats
policy: torch.nn.Module | None = None
eagle_proc = None
STATS: dict[str, dict[str, Any]] | None = None


# --------------------------------------------------------------------------------------
# Yardımcı fonksiyonlar – GrootPackInputsStep + Eagle encode taklidi
# --------------------------------------------------------------------------------------


def load_stats_from_hf() -> dict[str, dict[str, Any]]:
    """Hugging Face dataset'ten `meta/stats.json` indirip yükle."""
    path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="meta/stats.json",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats


def _align_vec(vec: Any, target_dim: int, *, default: float) -> torch.Tensor:
    """LeRobot GrootPackInputsStep._align_vec ile aynı mantık."""
    t = torch.as_tensor(vec, dtype=torch.float32)
    t = t.flatten()
    d = int(t.shape[-1]) if t.numel() > 0 else 0
    if d == target_dim:
        return t
    if d < target_dim:
        pad = torch.full((target_dim - d,), default, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)
    return t[:target_dim]


def _min_max_norm(x: torch.Tensor, key: str) -> torch.Tensor:
    """LeRobot GrootPackInputsStep._min_max_norm ile aynı mantık."""
    if STATS is None or key not in STATS:
        return x
    stats_k = STATS[key]
    last_dim = x.shape[-1]
    min_v = _align_vec(stats_k.get("min", torch.zeros(last_dim)), last_dim, default=0.0).to(
        x.device
    )
    max_v = _align_vec(stats_k.get("max", torch.ones(last_dim)), last_dim, default=1.0).to(
        x.device
    )
    denom = max_v - min_v
    mask = denom != 0
    safe_denom = torch.where(mask, denom, torch.ones_like(denom))
    mapped = 2 * (x - min_v) / safe_denom - 1
    return torch.where(mask, mapped, torch.zeros_like(mapped))


def preprocess_image_to_eagle(img_pil: Image.Image) -> dict[str, torch.Tensor]:
    """
    GrootEagleEncodeStep + GrootEagleCollateStep taklidi:
      - Tek görüntü + varsayılan dil prompt'u ile conv hazırla
      - AutoProcessor ile eagle_* tensörlerini üret
    """
    lang = "Perform the task."
    lang_formatted = str([lang])
    text_content = [{"type": "text", "text": lang_formatted}]
    image_content = [{"type": "image", "image": img_pil}]
    conv = [{"role": "user", "content": image_content + text_content}]

    text_list = [
        eagle_proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    ]
    img_inputs, vid_inputs = eagle_proc.process_vision_info(conv)

    eagle_inputs = eagle_proc(
        text=text_list,
        images=img_inputs,
        images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
        return_tensors="pt",
        padding=True,
    )

    result: dict[str, torch.Tensor] = {}
    for k, v in eagle_inputs.items():
        result[f"eagle_{k}"] = v
    return result


def preprocess_state(state_vec: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GrootPackInputsStep için state hazırlığı:

    Giriş:
      - state_vec: 17 boyutlu observation.state (normalize edilmemiş)
    Çıkış:
      - state: (1, 1, MAX_STATE_DIM)
      - state_mask: (1, 1, MAX_STATE_DIM)
    """
    x = torch.tensor(state_vec, dtype=torch.float32)  # (D,)
    x = _min_max_norm(x.unsqueeze(0), STATE_FEATURE).squeeze(0)  # (D,) -> normalize
    d = x.shape[0]

    if d > MAX_STATE_DIM:
        x = x[:MAX_STATE_DIM]
        d = MAX_STATE_DIM
    elif d < MAX_STATE_DIM:
        pad = torch.zeros(MAX_STATE_DIM - d, dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)

    x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, MAX_STATE_DIM)
    state_mask = torch.zeros(1, 1, MAX_STATE_DIM, dtype=torch.bool)
    state_mask[:, :, :d] = True
    return x, state_mask


def postprocess_action(action: torch.Tensor) -> list[float]:
    """
    GrootActionUnpackUnnormalizeStep benzeri:
      - (B, T, D) veya (B, D) -> (17,) R1 action
      - ACTION feature'ı için dataset stats ile inverse MIN_MAX
    """
    if STATS is None or ACTION_FEATURE not in STATS:
        return action.squeeze(0).cpu().numpy().tolist()[:17]

    if action.dim() == 3:
        # (B, T, D) -> ilk timestep
        action = action[:, 0, :]
    # (B, D) – son boyutu modelin gerçekten ürettiği D kadar kullan
    last_dim = action.shape[-1]

    stats_k = STATS[ACTION_FEATURE]
    min_v = _align_vec(
        stats_k.get("min", torch.zeros(last_dim)),
        last_dim,
        default=0.0,
    ).to(action.device)
    max_v = _align_vec(
        stats_k.get("max", torch.ones(last_dim)),
        last_dim,
        default=1.0,
    ).to(action.device)
    denom = max_v - min_v
    mask = denom != 0
    safe_denom = torch.where(mask, denom, torch.ones_like(denom))

    inv = (action + 1.0) * 0.5 * safe_denom + min_v
    action = torch.where(mask, inv, min_v)

    # İlk 17 boyutu kullan (dataset'in gerçek action dim'i)
    return action.squeeze(0).cpu().numpy().tolist()[:17]


# --------------------------------------------------------------------------------------
# WebSocket handler
# --------------------------------------------------------------------------------------


async def handle_client(websocket, *args, **kwargs):
    print("New connection.")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)

                # 1) State parse – 17 boyutlu bekliyoruz
                state_vec = data.get("state", None)
                if not isinstance(state_vec, list) or len(state_vec) != 17:
                    await websocket.send(
                        json.dumps(
                            {"error": "state must be a list of 17 floats (R1 observation.state)."}
                        )
                    )
                    continue

                # 2) Image parse (Base64 JPEG -> PIL)
                image_b64 = data.get("image", "")
                if not image_b64:
                    await websocket.send(json.dumps({"error": "Missing image field."}))
                    continue

                try:
                    image_bytes = base64.b64decode(image_b64)
                    np_arr = np.frombuffer(image_bytes, np.uint8)
                    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        raise ValueError("cv2.imdecode returned None")
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                except Exception as e:
                    await websocket.send(json.dumps({"error": f"Failed to decode image: {e}"}))
                    continue

                # 3) Eagle + state preprocess
                eagle_batch = preprocess_image_to_eagle(img_pil)
                state, state_mask = preprocess_state(state_vec)

                batch: dict[str, Any] = {}
                for k, v in eagle_batch.items():
                    batch[k] = v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                batch["state"] = state.to(DEVICE)
                batch["state_mask"] = state_mask.to(DEVICE)
                batch["embodiment_id"] = torch.tensor([EMBODIMENT_ID], dtype=torch.long).to(DEVICE)

                # 4) GR00T policy ile action seç
                with torch.no_grad():
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")
                    ):
                        action = policy.select_action(batch)

                # 5) Inverse normalize + slice
                action_list = postprocess_action(action)

                await websocket.send(json.dumps({"action": action_list}))

            except Exception as e:
                import traceback

                traceback.print_exc()
                await websocket.send(json.dumps({"error": str(e)}))
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


async def main():
    global policy, eagle_proc, STATS

    print(f"Loading dataset stats for {DATASET_REPO_ID}@{DATASET_REVISION} ...")
    STATS = load_stats_from_hf()
    print("Stats keys:", list(STATS.keys()))

    print(f"Loading GR00T policy from {PRETRAINED_POLICY_PATH} ...")
    cfg = PreTrainedConfig.from_pretrained(PRETRAINED_POLICY_PATH)
    cfg.device = DEVICE
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(PRETRAINED_POLICY_PATH, config=cfg)
    policy.to(DEVICE)
    policy.eval()
    print(f"Loaded policy type: {cfg.type}")

    cache_dir = HF_LEROBOT_HOME / "lerobot/eagle2hg-processor-groot-n1p5"
    eagle_proc = AutoProcessor.from_pretrained(
        str(cache_dir),
        trust_remote_code=True,
        use_fast=True,
    )
    eagle_proc.tokenizer.padding_side = "left"
    print("Eagle processor ready.")

    print("Starting WebSocket Server on port 8765...")
    server = await websockets.serve(handle_client, "0.0.0.0", 8765)
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())

