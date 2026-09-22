"""Command line. There is no order command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aegis.config import laboratory_settings
from aegis.experiment import run_holdout_once, run_research, write_experiment
from aegis.report import write_site
from aegis.safety import HoldoutLockError, assert_research_only
from aegis.store import load_dataset, write_dataset
from aegis.synthetic import DATASET_ID, generate_laboratory_eurusd


def main(argv: list[str] | None = None) -> int:
    assert_research_only()
    parser = argparse.ArgumentParser(prog="aegis", description="Research-only FX experiment engine.")
    parser.add_argument("--root", default=".", help="Repository root. Defaults to the current directory.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("lab-run", help="Run the pre-registered synthetic laboratory experiment and write the bulletin.")
    audit = sub.add_parser("audit", help="Print the audit block of a saved experiment.")
    audit.add_argument("experiment_id")
    holdout = sub.add_parser("holdout-once", help="Open the locked year a single time. Refuses a second call.")
    holdout.add_argument("--confirm-once", action="store_true", help="Required. Acknowledges that this look cannot be repeated.")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.command == "lab-run":
        return lab_run(root)
    if args.command == "audit":
        return print_audit(root, args.experiment_id)
    if args.command == "holdout-once":
        if not args.confirm_once:
            print("Refusing. Pass --confirm-once. The locked year can be scored only once for a settings hash.", file=sys.stderr)
            return 2
        return holdout_once(root)
    print("Unknown command.", file=sys.stderr)
    return 2


def lab_run(root: Path) -> int:
    settings = laboratory_settings()
    table, manifest = _laboratory_table(root)
    print(f"dataset {manifest['dataset_id']} rows {len(table)} synthetic {manifest.get('synthetic')}", flush=True)
    result = run_research(table, manifest, settings)
    destination = write_experiment(root / "experiments", result)
    write_site(result, root / "research" / "bulletin.html", root / "research" / "latest.json")
    verdict = result["verdict"]
    print(f"wrote {destination}")
    print("data archive only; index.html was not changed")
    print(f"laboratory_process_verdict {verdict['laboratory_process_verdict']}")
    print(f"market_classification {verdict['market_classification']}")
    print(f"audit_valid {result['audit']['valid']}")
    print("live_execution_authorized false")
    return 0 if result["audit"]["valid"] else 1


def holdout_once(root: Path) -> int:
    settings = laboratory_settings()
    table, manifest = _laboratory_table(root)
    try:
        result = run_holdout_once(table, manifest, settings, root / "experiments" / "holdout_locks")
    except HoldoutLockError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out = root / "experiments" / "holdout_locks" / f"{result['settings_sha256']}_result.json"
    public = {key: value for key, value in result.items() if key not in {"predictions", "trade_rows"}}
    out.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
    print(f"holdout scored once: trades {result['trades']} net_pips {result['net_pips']:.4f}")
    print("This result is not part of the walk-forward equity curve and does not authorize live execution.")
    return 0


def print_audit(root: Path, experiment_id: str) -> int:
    path = root / "experiments" / experiment_id / "experiment.json"
    if not path.exists():
        print(f"No experiment at {path}", file=sys.stderr)
        return 2
    payload = json.loads(path.read_text())
    print(json.dumps(payload.get("audit"), indent=2, sort_keys=True))
    return 0 if payload.get("audit", {}).get("valid") else 1


def _laboratory_table(root: Path):
    data_root = root / "data" / "sets"
    manifest_path = data_root / DATASET_ID / "manifest.json"
    if not manifest_path.exists():
        table, meta = generate_laboratory_eurusd()
        write_dataset(data_root, DATASET_ID, table, meta)
    table, manifest = load_dataset(data_root, DATASET_ID)
    return table, manifest
