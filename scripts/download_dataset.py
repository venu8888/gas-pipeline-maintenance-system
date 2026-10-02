"""Download the public Kaggle dataset without placing it under source control."""
from __future__ import annotations
import argparse
from pathlib import Path

DATASET = "simplexitypipeline/pipeline-defect-dataset"

def download(destination: Path, dataset: str = DATASET) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
        source = Path(kagglehub.dataset_download(dataset))
        print(f"Dataset downloaded/cached at {source}")
        print(f"Copy or configure dataset.root to that location; requested destination is {destination}")
        return source
    except ImportError:
        pass
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        api.dataset_download_files(dataset, path=str(destination), unzip=True)
        return destination
    except ImportError as exc:
        raise RuntimeError("Install kagglehub or kaggle and configure Kaggle credentials") from exc
    except Exception as exc:
        raise RuntimeError(f"Kaggle download failed: {exc}") from exc

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--destination", type=Path, default=Path("data/raw")); p.add_argument("--dataset", default=DATASET)
    args=p.parse_args(); print(download(args.destination,args.dataset))
