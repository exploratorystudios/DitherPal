"""Shared, local neural-depth inference and bounded source-image caching."""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import urllib.request

import cv2
import numpy as np
from PIL import Image

MODEL_URL = ('https://huggingface.co/onnx-community/depth-anything-v2-small/'
             'resolve/f7421df/onnx/model.onnx')
MODEL_SHA256 = 'afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c'
DEFAULT_MODEL = Path(os.environ.get('DITHERPAL_MODEL_DIR', Path(__file__).resolve().parent.parent/'models')) / 'depth-anything-v2-small.onnx'
_lock = threading.RLock()
_sessions = {}
_verified = {}
_files = OrderedDict()


def source_key(path):
    try:
        file = Path(path).resolve()
        stat = file.stat()
        return str(file), stat.st_mtime_ns, stat.st_size
    except (OSError, TypeError):
        return None


def download_model(destination=DEFAULT_MODEL, progress=None):
    """Verify cached weights, or download a pinned model with atomic installation."""
    destination = Path(destination)
    with _lock:
        if destination.is_file():
            signature = source_key(destination)
            if _verified.get(str(destination)) != signature:
                digest = hashlib.sha256()
                with destination.open('rb') as existing:
                    for chunk in iter(lambda: existing.read(1024*1024), b''):
                        digest.update(chunk)
                if digest.hexdigest() != MODEL_SHA256:
                    raise ValueError(f'Depth model checksum failed. Remove {destination} and retry.')
                _verified[str(destination)] = signature
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix='depth-', suffix='.download', dir=destination.parent)
        temporary = Path(name)
        digest = hashlib.sha256()
        try:
            request = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'DitherPal/NeuralDepth'})
            with os.fdopen(descriptor, 'wb') as output, urllib.request.urlopen(request, timeout=120) as response:
                downloaded = 0
                if progress:
                    progress('Downloading neural depth model (about 99 MB, once)…')
                while chunk := response.read(1024*1024):
                    digest.update(chunk)
                    output.write(chunk)
                    downloaded += len(chunk)
                    if progress and downloaded//(10*1024*1024) != (downloaded-len(chunk))//(10*1024*1024):
                        progress(f'Downloading neural depth model: {downloaded/1024**2:.0f} MiB')
            if digest.hexdigest() != MODEL_SHA256:
                raise ValueError('Downloaded depth model failed checksum verification')
            temporary.replace(destination)
            _verified[str(destination)] = source_key(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def preprocess(rgb, input_size=518, fixed_shape=None):
    height, width = rgb.shape[:2]
    if fixed_shape is not None:
        target_h, target_w = fixed_shape
    else:
        scale = min((input_size/height, input_size/width), key=lambda value: abs(1-value))
        target_h = max(14, round(height*scale/14)*14)
        target_w = max(14, round(width*scale/14)*14)
    image = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)/255
    image = (image-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None])


def infer_depth(rgb, model_path=DEFAULT_MODEL, threads=4, progress=None):
    """One persistent CPU session; serialized inference avoids nested thread-pool overload."""
    import onnxruntime as ort
    ort.disable_telemetry_events()

    with _lock:
        key = str(Path(model_path).resolve()), threads
        if key not in _sessions:
            path = download_model(model_path, progress)
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            _sessions[key] = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        session = _sessions[key]
        specification = session.get_inputs()[0]
        shape = specification.shape
        fixed = tuple(shape[-2:]) if all(isinstance(n, int) and n > 0 for n in shape[-2:]) else None
        prediction = np.asarray(session.run(None, {specification.name: preprocess(rgb, fixed_shape=fixed)})[0]).squeeze()
    if prediction.ndim != 2 or not np.isfinite(prediction).all():
        raise ValueError('Neural depth returned an invalid map')
    return cv2.resize(prediction.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)


def depth_to_z(prediction, near=1., far=4.):
    if not 0 < near < far or prediction.ndim != 2 or not np.isfinite(prediction).all():
        raise ValueError('Expected finite depth and 0 < near < far')
    low, high = np.percentile(prediction, [2, 98])
    normalized = (np.full_like(prediction, .5, dtype=np.float32) if high-low < 1e-6
                  else np.clip((prediction-low)/(high-low), 0, 1).astype(np.float32))
    return (1/(1/far+normalized*(1/near-1/far))).astype(np.float32), normalized


def scale_depth(near, strength=100):
    if not 0 <= strength <= 200:
        raise ValueError('Depth strength must be between 0 and 200 percent')
    return np.ascontiguousarray(np.clip((near-.5)*(strength/100.)+.5, 0, 1), dtype=np.float32)


def predict_near(rgb, progress=None):
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    elif rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    rgb = np.clip(rgb[..., :3], 0, 255).astype(np.uint8)
    height, width = rgb.shape[:2]
    scale = min(1., 640/max(height, width))
    small = cv2.resize(rgb, (max(1, round(width*scale)), max(1, round(height*scale))), interpolation=cv2.INTER_AREA)
    if progress:
        progress('Estimating neural depth…')
    _, near = depth_to_z(infer_depth(small, progress=progress))
    return near


def source_image(path):
    """First-frame preview for media; original RGB for photographs, capped for caching."""
    path = Path(path)
    if path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}:
        capture = cv2.VideoCapture(str(path))
        try:
            ok, bgr = capture.read()
            if not ok:
                raise ValueError('Could not read a video frame')
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
    else:
        with Image.open(path) as original:
            image = original.convert('RGB')
    image.thumbnail((640, 640), Image.Resampling.LANCZOS)
    return np.array(image)


def image_and_depth(path, progress=None):
    key = source_key(path)
    if key is None:
        raise FileNotFoundError(f'Source unavailable: {path}')
    with _lock:
        if key not in _files:
            rgb = source_image(path)
            near = predict_near(rgb, progress)
            rgb.setflags(write=False)
            near.setflags(write=False)
            _files[key] = rgb, near
            while len(_files) > 4:
                _files.popitem(last=False)
        _files.move_to_end(key)
        return _files[key]


def depth_for_file(path, progress=None):
    return image_and_depth(path, progress)[1]
