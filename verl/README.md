# Bundled verl Framework

This directory contains the framework snapshot used by the public GC-OPD
release. The retained policy methods are Raw/GRPO, OPD, ExOPD, Uni-OPD,
PowerOPD, FiRe-OPD, and GC-OPD. It is packaged locally so the focused CPU tests
and training entrypoints resolve the same implementation.

The upstream project is `verl`, version `0.6.1`, distributed under Apache-2.0.
See `LICENSE` and `Notice.txt` for attribution and licensing information.

Install for CPU verification without resolving optional GPU dependencies:

```bash
python -m pip install -e . --no-deps
```

Full training additionally requires a compatible PyTorch, Ray, Transformers,
and vLLM or SGLang environment. Those binary dependencies are intentionally not
vendored in this source archive.
