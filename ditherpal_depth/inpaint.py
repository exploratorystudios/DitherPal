"""Local LaMa inpainting for the hidden layers behind foreground depth steps."""
import threading

import cv2
import numpy as np

from .model import DEFAULT_MODEL, download_model

LAMA_URL = ('https://huggingface.co/Carve/LaMa-ONNX/'
            'resolve/c3c0c9e468934d62e79c329e35d82dd09ff8c444/lama_fp32.onnx')
LAMA_SHA256 = '1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6'
LAMA_MODEL = DEFAULT_MODEL.parent/'lama-fp32.onnx'
LAMA_SIZE = 512   # The ONNX export has a fixed square input.
_lock = threading.RLock()
_sessions = {}


def _session(model_path, threads, progress):
    import onnxruntime as ort
    ort.disable_telemetry_events()

    key = str(model_path), threads
    if key not in _sessions:
        path = download_model(model_path, progress, LAMA_URL, LAMA_SHA256, 'LaMa inpainting model', 'about 208 MB')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        _sessions[key] = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    return _sessions[key]


def inpaint(rgb, mask, model_path=LAMA_MODEL, threads=4, progress=None):
    """Paint the masked pixels of an RGB image; pixels outside the mask are returned unchanged."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return rgb.copy()
    height, width = mask.shape
    side = max(height, width)
    # Reflect-pad to a square so the fixed input does not stretch the artwork.
    image = np.pad(rgb, ((0, side-height), (0, side-width), (0, 0)), mode='reflect')
    hole = np.pad(mask, ((0, side-height), (0, side-width)))
    image = cv2.resize(image, (LAMA_SIZE,)*2, interpolation=cv2.INTER_AREA if side > LAMA_SIZE else cv2.INTER_CUBIC)
    # Area-resized masks keep thin holes that nearest-neighbour sampling would drop.
    hole = cv2.resize(np.float32(hole), (LAMA_SIZE,)*2, interpolation=cv2.INTER_AREA) > 0
    with _lock:
        session = _session(model_path, threads, progress)
        inputs = session.get_inputs()
        painted = session.run(None, {inputs[0].name: (image.astype(np.float32)/255).transpose(2, 0, 1)[None],
                                     inputs[1].name: np.float32(hole)[None, None]})[0]
    painted = np.clip(painted[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
    painted = cv2.resize(painted, (side, side), interpolation=cv2.INTER_CUBIC)[:height, :width]
    result = rgb.copy()
    result[mask] = painted[mask]
    return result
