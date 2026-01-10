"""Hugging Face Hub integration for models."""
import os
from pathlib import Path
from typing import Optional, Type
import torch
from pytorch_lightning import LightningModule

try:
    from huggingface_hub import HfApi, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

MODEL_REPO = "LookUpMark/DYLEM-GRID-models"
CACHE_DIR = os.path.expanduser("~/.cache/dylem-grid/models")
MODEL_FILES = {"bilstm": "bilstm_best.ckpt", "transformer": "transformer_best.ckpt"}


def save_to_hub(model: LightningModule, model_name: str = "bilstm", repo_id: str = MODEL_REPO, token: Optional[str] = None) -> str:
    """Save model to Hugging Face Hub."""
    if not HF_AVAILABLE: raise ImportError("pip install huggingface_hub")
    api = HfApi(token=token)
    try: api.create_repo(repo_id, exist_ok=True)
    except: pass
    
    local = Path(CACHE_DIR) / f"{model_name}_upload.ckpt"
    local.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hparams": dict(model.hparams)}, local)
    
    url = api.upload_file(str(local), MODEL_FILES.get(model_name, f"{model_name}.ckpt"), repo_id, commit_message=f"Upload {model_name}")
    local.unlink()
    print(f"Uploaded to {repo_id}")
    return url


def load_from_hub(model_class: Type[LightningModule], model_name: str = "bilstm", repo_id: str = MODEL_REPO,
                  local_path: Optional[str] = None, force_download: bool = False, **kwargs) -> LightningModule:
    """Load model from local or Hub (with fallback)."""
    filename = MODEL_FILES.get(model_name, f"{model_name}.ckpt")
    
    # Try local path first
    if local_path and os.path.exists(local_path):
        print(f"Loading from local: {local_path}")
        return _load_ckpt(model_class, local_path, **kwargs)
    
    # Try cache
    cached = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cached) and not force_download:
        print(f"Loading from cache: {cached}")
        return _load_ckpt(model_class, cached, **kwargs)
    
    # Download from Hub
    if not HF_AVAILABLE: raise ImportError("pip install huggingface_hub")
    print(f"Downloading from Hub: {repo_id}/{filename}")
    path = hf_hub_download(repo_id, filename, cache_dir=CACHE_DIR, force_download=force_download)
    return _load_ckpt(model_class, path, **kwargs)


def _load_ckpt(model_class, path, **kwargs):
    ckpt = torch.load(path, map_location="cpu")
    if "hparams" in ckpt:
        hparams = {**ckpt["hparams"], **kwargs}
        model = model_class(**hparams)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = model_class.load_from_checkpoint(path, **kwargs)
    return model


def get_model(model_name: str = "bilstm", local_dir: str = "models/checkpoints", **kwargs) -> LightningModule:
    """Get model - local if exists, else from Hub."""
    from src.models import BiLSTMModule, TransformerModule
    cls = {"bilstm": BiLSTMModule, "transformer": TransformerModule}[model_name]
    local = os.path.join(local_dir, MODEL_FILES.get(model_name))
    return load_from_hub(cls, model_name, local_path=local, **kwargs)


def push_all_models(checkpoint_dir: str = "models/checkpoints", repo_id: str = MODEL_REPO, token: Optional[str] = None):
    """Push all models to Hub."""
    if not HF_AVAILABLE: raise ImportError("pip install huggingface_hub")
    api = HfApi(token=token)
    try: api.create_repo(repo_id, exist_ok=True)
    except: pass
    
    for name, fname in MODEL_FILES.items():
        path = os.path.join(checkpoint_dir, fname)
        if os.path.exists(path):
            api.upload_file(path, fname, repo_id, commit_message=f"Upload {name}")
            print(f"Uploaded {name}")
