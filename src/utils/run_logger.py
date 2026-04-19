import json
from datetime import datetime
from pathlib import Path


class RunLogger:
    def __init__(self, root_dir, prefix):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_prefix = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in prefix)
        self.file_path = self.root_dir / f"{timestamp}_{safe_prefix}.jsonl"

    def log(self, event, payload=None):
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "payload": payload or {},
        }
        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def path(self):
        return str(self.file_path)

