"""Global determinism: seeds + versions manifest."""
import os, random, json, platform, sys
import numpy as np

def set_seed(seed: int = 1337):
    seed = int(os.environ.get("ELPB_SEED", seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    return seed

def versions_manifest():
    out = {"python": sys.version, "platform": platform.platform()}
    for pkg in ["numpy", "torch", "transformers", "datasets", "sklearn", "scipy"]:
        try:
            m = __import__(pkg)
            out[pkg] = getattr(m, "__version__", "ok")
        except Exception as e:
            out[pkg] = f"missing: {e}"
    return out
