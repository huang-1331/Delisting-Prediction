"""Count legacy CSV widths without printing or changing financial records."""

from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json

from .collection_runner import ProgressLog
from .config import RAW_DATA_DIR, PROJECT_ROOT
from .collection_state import file_sha256


def main() -> int:
    """Tee an actual structural audit of legacy files; emit aggregate widths only."""
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_%f")
    path = PROJECT_ROOT / "reports/monthly" / f"task02_legacy_schema_{stamp}.json"
    log = ProgressLog(path.with_suffix(".txt"))
    try:
        result = []
        for source in sorted(RAW_DATA_DIR.glob("dart_annual_statement*.csv")):
            with source.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader)
                widths = Counter(len(row) for row in reader)
            result.append({"path": source.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": file_sha256(source), "header_fields": len(header),
                "row_width_counts": dict(widths), "mismatched_width_rows": sum(count for width, count in widths.items() if width != len(header))})
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
