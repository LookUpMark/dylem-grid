"""LightningDataModule for DYLEM-GRID with Hugging Face Hub support."""
from __future__ import annotations
import os, random
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

HF_REPO = "LookUpMark/DYLEM-GRID"
CACHE_DIR = os.path.expanduser("~/.cache/dylem-grid")


class GestureDataModule(LightningDataModule):
    """DataModule for DYLEM-GRID. Auto-downloads from HF Hub if data_path=None."""
    
    def __init__(self, data_path: Optional[str] = None, data_type: str = "Cleaned",
                 batch_size: int = 32, pca_variance: float = 0.95, val_split: float = 0.2,
                 test_split: float = 0.0, cv_fold: Optional[int] = None, n_folds: int = 5,
                 seed: int = 42, num_workers: int = 4, force_download: bool = False,
                 balance_classes: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.data_path, self.data_type = data_path, data_type
        self.batch_size, self.pca_variance = batch_size, pca_variance
        self.val_split, self.test_split = val_split, test_split
        self.cv_fold, self.n_folds, self.seed = cv_fold, n_folds, seed
        self.num_workers, self.force_download = num_workers, force_download
        self.balance_classes = balance_classes
        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.label_encoder, self.input_size, self.num_classes = None, 0, 0
        self.class_names, self._resolved_path = [], None

    def prepare_data(self):
        if self.data_path:
            self._resolved_path = self.data_path
            return
        cache_path = os.path.join(CACHE_DIR, "DYLEM-GRID")
        if os.path.exists(cache_path) and not self.force_download:
            self._resolved_path = cache_path
            return
        if not HF_AVAILABLE:
            raise ImportError("pip install huggingface_hub")
        print(f"Downloading dataset from {HF_REPO}...")
        self._resolved_path = snapshot_download(HF_REPO, repo_type="dataset", 
                                                 local_dir=cache_path, local_dir_use_symlinks=False)

    def setup(self, stage: Optional[str] = None):
        if not self._resolved_path: self.prepare_data()
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        data, labels = self._load_data()
        data, labels = self._preprocess(data, labels)
        data, labels = self._apply_pca(data, labels)
        X, y = self._to_tensors(data, labels)
        self.input_size, self.num_classes = X.shape[2], len(self.label_encoder.classes_)
        self.class_names = list(self.label_encoder.classes_)
        self._setup_cv_split(X, y) if self.cv_fold is not None else self._setup_simple_split(X, y)

    def _load_data(self):
        lst, labels = [], []
        base = os.path.join(self._resolved_path, f"DYLEM-GRID_{self.data_type}")
        if not os.path.exists(base): base = self._resolved_path
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".csv"):
                    try:
                        lst.append(pd.read_csv(os.path.join(root, f)))
                        labels.append(os.path.basename(os.path.dirname(os.path.join(root, f))))
                    except: pass
        if not lst: raise ValueError(f"No CSV files in {base}")
        return lst, labels

    def _preprocess(self, data, labels):
        for df in data: df.bfill(inplace=True)
        data = [df.loc[:, ~df.T.duplicated(keep="first")] for df in data]
        lengths = [len(df) for df in data]
        concat = pd.concat(data, ignore_index=True)
        concat["_lbl"] = sum([[l]*n for l, n in zip(labels, lengths)], [])
        # Drop low-variance columns
        cols = [c for c in concat.columns if c != "_lbl"]
        keep = [c for c in cols if concat[c].value_counts(normalize=True).max() < 0.9] + ["_lbl"]
        concat = concat[keep]
        # Outlier removal per class
        for lbl in concat["_lbl"].unique():
            mask = concat["_lbl"] == lbl
            for col in [c for c in concat.columns if c != "_lbl" and pd.api.types.is_numeric_dtype(concat[c])]:
                if pd.api.types.is_integer_dtype(concat[col]): concat[col] = concat[col].astype(float)
                mean, std = concat.loc[mask, col].mean(), concat.loc[mask, col].std()
                if std and std > 0:
                    outliers = mask & ((concat[col] - mean).abs() > 3 * std)
                    concat.loc[outliers, col] = mean
        # Normalize
        num_cols = [c for c in concat.select_dtypes(include=[np.number]).columns if c != "_lbl"]
        if num_cols: concat[num_cols] = MinMaxScaler().fit_transform(concat[num_cols])
        concat = concat.drop("_lbl", axis=1)
        # Split back
        proc, idx = [], 0
        for n in lengths: proc.append(concat.iloc[idx:idx+n].copy()); idx += n
        combined = list(zip(proc, labels)); random.shuffle(combined)
        
        if self.balance_classes:
            combined = self._balance_data(combined)
            
        return list(zip(*combined))

    def _balance_data(self, combined_data):
        """Oversample minority classes to match majority class count."""
        from collections import defaultdict
        by_label = defaultdict(list)
        for item in combined_data:
            by_label[item[1]].append(item)
            
        max_count = max(len(items) for items in by_label.values())
        balanced = []
        for label, items in by_label.items():
            if len(items) < max_count:
                # Oversample
                extras = random.choices(items, k=max_count - len(items))
                balanced.extend(items + extras)
            else:
                balanced.extend(items)
        random.shuffle(balanced)
        return balanced

    def _apply_pca(self, data, labels):
        lengths = [len(df) for df in data]
        concat = pd.concat(data, ignore_index=True)
        num_cols = concat.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0: return data, labels
        transformed = PCA(n_components=self.pca_variance).fit_transform(concat[num_cols])
        df = pd.DataFrame(transformed, columns=[f"PC{i+1}" for i in range(transformed.shape[1])])
        result, idx = [], 0
        for n in lengths: result.append(df.iloc[idx:idx+n].copy()); idx += n
        return result, labels

    def _to_tensors(self, data, labels):
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        max_len, n_feat = max(len(df) for df in data), data[0].shape[1]
        padded = []
        for df in data:
            arr = df.values
            if len(arr) < max_len: arr = np.vstack([arr, np.zeros((max_len - len(arr), n_feat))])
            padded.append(arr)
        return torch.FloatTensor(np.array(padded, dtype=np.float32)), torch.LongTensor(y)

    def _setup_simple_split(self, X, y):
        if self.test_split > 0:
            X, X_t, y, y_t = train_test_split(X, y, test_size=self.test_split, random_state=self.seed, stratify=y)
            self.test_dataset = TensorDataset(X_t, y_t)
        X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=self.val_split, random_state=self.seed, stratify=y)
        self.train_dataset, self.val_dataset = TensorDataset(X_tr, y_tr), TensorDataset(X_v, y_v)

    def _setup_cv_split(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for i, (tr, va) in enumerate(skf.split(X, y)):
            if i == self.cv_fold:
                self.train_dataset = TensorDataset(X[tr], y[tr])
                self.val_dataset = TensorDataset(X[va], y[va])
                break

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        if not self.test_dataset: raise ValueError("No test set. Set test_split > 0")
        return DataLoader(self.test_dataset, self.batch_size, num_workers=self.num_workers, pin_memory=True)
