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

Weights are downloaded separately, verified before use, and excluded from Git.
No model weights or training datasets are distributed in this repository.
Source images stay on the local machine. The only network request required for
inference is the initial download of these public weights.
