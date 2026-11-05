import hashlib
import json
from pathlib import Path
from typing import Any


def video_id(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


class WorkDir:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / name

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def write_json(self, name: str, data: Any) -> None:
        self.path(name).write_text(json.dumps(data, indent=2))

    def read_json(self, name: str) -> Any:
        return json.loads(self.path(name).read_text())

    def record(self, stage: str, status: str, seconds: float, warnings: list[str]) -> None:
        run = self.read_json("run.json") if self.has("run.json") else {"stages": {}}
        run["stages"][stage] = {"status": status, "seconds": round(seconds, 2), "warnings": list(warnings)}
        self.write_json("run.json", run)
        lines = [f"status: {status}", f"seconds: {seconds:.2f}", *[f"warning: {w}" for w in warnings]]
        self.path(f"{stage}.log").write_text("\n".join(lines) + "\n")
