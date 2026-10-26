# Neural depth model

DitherPal uses Depth Anything V2 Small for local relative-depth estimation.

- Original project: https://github.com/DepthAnything/Depth-Anything-V2
- Model: https://huggingface.co/depth-anything/Depth-Anything-V2-Small
- ONNX conversion: https://huggingface.co/onnx-community/depth-anything-v2-small
- Pinned conversion revision: `f7421df`
- File: `onnx/model.onnx`, approximately 99 MB
- SHA-256: `afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c`

The Small model and the linked ONNX conversion are published under Apache-2.0.
See their model cards and the original project's license for the applicable terms
and attribution. Other model sizes can have different licenses and are not used
by this application.

## Inpainting model

Neural 3D animation uses LaMa (big-lama) to paint hidden layers behind foreground
depth steps.

- Original project: https://github.com/advimman/lama
- ONNX export: https://huggingface.co/Carve/LaMa-ONNX
- Pinned export revision: `c3c0c9e468934d62e79c329e35d82dd09ff8c444`
- File: `lama_fp32.onnx` (saved as `lama-fp32.onnx`), approximately 208 MB, fixed 512×512 input
- SHA-256: `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6`

LaMa and the ONNX export are published under Apache-2.0.

## Distribution

Weights are downloaded separately, verified before use, and excluded from Git.
No model weights or training datasets are distributed in this repository.
Source images stay on the local machine. The only network request required for
inference is the initial download of these public weights.
