import clip
import torch

_clip_cache = {}

def load_clip_cached(model_name: str, device: str):
    key = (model_name, device)
    if key not in _clip_cache:
        print(f"[CLIP Loader] Loading CLIP model {model_name} on {device}...")
        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        _clip_cache[key] = (model, preprocess)
    else:
        print(f"[CLIP Loader] Using cached CLIP model {model_name} on {device}.")
    return _clip_cache[key]
