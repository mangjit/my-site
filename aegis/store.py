"""Immutable candle store. A dataset id is written once and then only read."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from aegis.safety import DataIntegrityError
from aegis.table import CandleTable
from aegis.timeutil import iso_z, parse_timestamp


MANIFEST = "manifest.json"
CANDLES = "candles.csv"

CSV_FIELDS = (
    "timestamp",
    "bid_o",
    "bid_h",
    "bid_l",
    "bid_c",
    "ask_o",
    "ask_h",
    "ask_l",
    "ask_c",
    "volume",
    "complete",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: float) -> str:
    return f"{value:.10f}"


def write_dataset(root: Path, dataset_id: str, table: CandleTable, meta: dict) -> Path:
    if not dataset_id or "/" in dataset_id or dataset_id.startswith("."):
        raise DataIntegrityError(f"Illegal dataset id: {dataset_id!r}")
    table.validate_clock()
    destination = root / dataset_id
    if destination.exists():
        raise DataIntegrityError(
            f"Dataset {dataset_id} already exists and is immutable. "
            "Write a new dataset id instead of replacing history."
        )
    destination.mkdir(parents=True)
    csv_path = destination / CANDLES
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i in range(len(table)):
            volume = table.volume[i]
            writer.writerow(
                {
                    "timestamp": iso_z(table.timestamp[i]),
                    "bid_o": _fmt(table.bid_o[i]),
                    "bid_h": _fmt(table.bid_h[i]),
                    "bid_l": _fmt(table.bid_l[i]),
                    "bid_c": _fmt(table.bid_c[i]),
                    "ask_o": _fmt(table.ask_o[i]),
                    "ask_h": _fmt(table.ask_h[i]),
                    "ask_l": _fmt(table.ask_l[i]),
                    "ask_c": _fmt(table.ask_c[i]),
                    "volume": "" if volume is None else _fmt(volume),
                    "complete": "1" if table.complete[i] else "0",
                }
            )
    manifest = {
        "dataset_id": dataset_id,
        "instrument": table.instrument,
        "granularity": table.granularity,
        "timestamp_meaning": table.timestamp_meaning,
        "rows": len(table),
        "start": iso_z(table.timestamp[0]) if table.timestamp else None,
        "end": iso_z(table.timestamp[-1]) if table.timestamp else None,
        "sha256": _sha256(csv_path),
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "immutable": True,
    }
    manifest.update(meta)
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def load_dataset(root: Path, dataset_id: str) -> tuple[CandleTable, dict]:
    destination = root / dataset_id
    manifest_path = destination / MANIFEST
    csv_path = destination / CANDLES
    if not manifest_path.exists() or not csv_path.exists():
        raise DataIntegrityError(f"Dataset {dataset_id} is missing a manifest or candle file.")
    manifest = json.loads(manifest_path.read_text())
    digest = _sha256(csv_path)
    if digest != manifest.get("sha256"):
        raise DataIntegrityError(
            f"Dataset {dataset_id} failed its sha256 check. "
            "Refusing to research on altered history."
        )
    table = CandleTable(
        instrument=manifest["instrument"],
        granularity=manifest["granularity"],
        timestamp=[],
        bid_o=[],
        bid_h=[],
        bid_l=[],
        bid_c=[],
        ask_o=[],
        ask_h=[],
        ask_l=[],
        ask_c=[],
        volume=[],
        complete=[],
        timestamp_meaning=manifest.get("timestamp_meaning", "bar_open_utc"),
    )
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            table.timestamp.append(parse_timestamp(row["timestamp"]))
            for name in ("bid_o", "bid_h", "bid_l", "bid_c", "ask_o", "ask_h", "ask_l", "ask_c"):
                getattr(table, name).append(float(row[name]))
            volume = row["volume"].strip()
            table.volume.append(None if volume == "" else float(volume))
            table.complete.append(row["complete"] == "1")
    if len(table) != manifest["rows"]:
        raise DataIntegrityError(
            f"Dataset {dataset_id} row count {len(table)} != manifest {manifest['rows']}."
        )
    table.validate_clock()
    return table, manifest
