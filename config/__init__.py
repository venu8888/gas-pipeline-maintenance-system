from pathlib import Path
import yaml

def load_config(path=None):
    path = Path(path or Path(__file__).with_name("config.yaml"))
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
