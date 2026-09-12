# Third-party components

ActivePour uses, but does not vendor or redistribute, these projects:

- [GeoTaichi](https://github.com/Yihao-Shi/GeoTaichi): GPL-3.0. It provides contact search, contact mechanics and integration. The adapter in `simulation/dem.py` configures and calls GeoTaichi; it is not an independently implemented DEM solver.
- [Taichi](https://github.com/taichi-dev/taichi): the simulation runtime, subject to its upstream license.
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers), Transformers, Accelerate and Safetensors: model loading and execution libraries, subject to their upstream licenses.
- [PyTorch](https://github.com/pytorch/pytorch): tensor and automatic-differentiation runtime, subject to its upstream license.
- [SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium): separately licensed model weights. Obtain them from the official model page and review its terms. This repository's code license does not grant rights to those weights.

No SD3.5 base weights, access tokens, third-party source tree, or private training checkpoints are included. Upstream dependencies and any combined redistribution remain subject to their applicable licenses. The public source release uses GPL-3.0-only; see `LICENSE`.
