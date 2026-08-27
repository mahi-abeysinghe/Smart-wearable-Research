"""Guided command-line helper for Smart Wear disease models.

This file does not train anything by itself. Run a specific subcommand such as
`python manage_models.py status` or `python manage_models.py train arrhythmia`.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile


BACKEND = Path(__file__).resolve().parent
DISEASES = {
    "arrhythmia": {
        "title": "Arrhythmia (MIT-BIH beat classification)",
        "dataset": BACKEND / "data" / "ECG Arrhythmia Classification Dataset.zip",
        "module": "AI_models.arrhythmia.train",
        "model": BACKEND / "AI_models" / "arrhythmia" / "saved_model" / "arrhythmia_model.joblib",
        "metrics": BACKEND / "AI_models" / "arrhythmia" / "saved_model" / "arrhythmia_model.metrics.json",
        "expected": ".csv",
        "sensor": "AD8232 ECG, sent as 250 Hz sample batches",
    },
    "wesad": {
        "title": "WESAD physiological stress classification",
        "dataset": BACKEND / "data" / "WESAD.zip",
        "module": "AI_models.wesad.train",
        "model": BACKEND / "AI_models" / "wesad" / "saved_model" / "wesad_model.joblib",
        "metrics": BACKEND / "AI_models" / "wesad" / "saved_model" / "wesad_model.metrics.json",
        "expected": ".pkl",
        "sensor": "live experimental subset uses MAX30102 heart rate and MLX90614 temperature only",
    },
    "parkinsons": {
        "title": "Parkinsonian wrist-motion pattern classification (offline only)",
        "dataset": BACKEND / "data" / "pads-parkinsons-disease-smartwatch-dataset-1.0.0.zip",
        "module": "AI_models.Parkinsons.train",
        "model": BACKEND / "AI_models" / "Parkinsons" / "saved_model" / "parkinsons_model.joblib",
        "metrics": BACKEND / "AI_models" / "Parkinsons" / "saved_model" / "parkinsons_model.metrics.json",
        "expected": ".json",
        "sensor": "requires wrist accelerometer; unavailable from the current four-sensor hardware",
    },
}


def selected(name):
    return DISEASES.items() if name == "all" else [(name, DISEASES[name])]


def dataset_check(name, config, verbose=True):
    path = config["dataset"]
    if not path.exists():
        if verbose:
            print(f"[MISSING] {name}: place the dataset at {path}")
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            files = [item for item in archive.namelist() if not item.endswith("/")]
    except zipfile.BadZipFile:
        print(f"[INVALID] {name}: {path.name} is not a valid ZIP file")
        return False
    matching = [item for item in files if item.lower().endswith(config["expected"])]
    if not matching:
        print(f"[INVALID] {name}: ZIP has no {config['expected']} files")
        return False
    if verbose:
        print(f"[READY] {name}: {path.name} ({len(matching)} {config['expected']} files)")
        for item in matching[:5]:
            print(f"        - {item}")
        if len(matching) > 5:
            print(f"        - ... and {len(matching) - 5} more")
    return True


def show_status(name):
    for key, config in selected(name):
        print(f"\n{config['title']}")
        dataset_check(key, config)
        print(f"[{'READY' if config['model'].exists() else 'MISSING'}] model: {config['model']}")
        if config["metrics"].exists():
            metrics = json.loads(config["metrics"].read_text(encoding="utf-8"))
            accuracy = metrics.get("accuracy")
            balanced = metrics.get("balanced_accuracy")
            summary = []
            if accuracy is not None:
                summary.append(f"accuracy={accuracy:.2%}")
            if balanced is not None:
                summary.append(f"balanced_accuracy={balanced:.2%}")
            print("[READY] metrics" + (f": {', '.join(summary)}" if summary else ""))
        else:
            print("[MISSING] metrics: train the model first")
        print(f"[INPUT] {config['sensor']}")


def inspect_dataset(name, config):
    """Safely show ZIP structure and CSV headers without training."""
    if not dataset_check(name, config, verbose=False):
        return False
    path = config["dataset"]
    print(f"\n{name.upper()}: {path}")
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.namelist() if not item.endswith("/")]
        matching = [item for item in members if item.lower().endswith(config["expected"])]
        print(f"Files in ZIP: {len(members)}; expected data files: {len(matching)}")
        for member in matching[:10]:
            print(f"  - {member}")
        if config["expected"] == ".pkl":
            print("WESAD pickle contents are parsed only by its trainer. Do not load pickle files from an untrusted source.")
            return True
        if config["expected"] == ".json":
            observation_member = next((member for member in matching if "/movement/observation_" in member), matching[0])
            observation = json.loads(archive.read(observation_member))
            first_session = (observation.get("session") or [{}])[0]
            first_record = (first_session.get("records") or [{}])[0]
            print(f"Example subject: {observation.get('subject_id', 'unknown')}")
            print(f"Source sampling rate: {observation.get('sampling_rate', 'unknown')} Hz")
            print(f"Example task: {first_session.get('record_name', 'unknown')}")
            print("Channels: " + ", ".join(first_record.get("channels", [])))
            print("The PADS trainer joins these observations with preprocessed/file_list.csv labels and reads referenced movement/timeseries TXT files.")
            return True
        for member in matching[:3]:
            with archive.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
                reader = csv.reader(text)
                header = next(reader, [])
                sample = next(reader, [])
            print(f"\nCSV: {member}")
            print(f"Columns ({len(header)}): {', '.join(header[:40])}" + (" ..." if len(header) > 40 else ""))
            likely_targets = [column for column in header if column.strip().lower() in {"type", "status", "label", "target", "class", "diagnosis", "parkinson", "parkinsons"}]
            print("Likely target column(s): " + (", ".join(likely_targets) if likely_targets else "NONE FOUND - loader may need editing"))
            if sample:
                print(f"First row has {len(sample)} values (values hidden; inspect the source dataset documentation for meanings).")
    return True


def show_guide():
    print("""
SMART WEAR MODEL GUIDE

1. Put each dataset ZIP in backend/data using the expected filename.
2. Check it without training:
       python manage_models.py check all
3. Inspect columns/structure before training:
       python manage_models.py inspect parkinsons
4. Train only one model at a time:
       python manage_models.py train arrhythmia
       python manage_models.py train wesad
       python manage_models.py train parkinsons
5. Review its held-out metrics:
       python manage_models.py metrics arrhythmia
6. Run offline integration tests:
       python manage_models.py test
7. Start Mosquitto, then start real-time inference:
       python manage_models.py serve

Use `status all` at any time. Training WESAD or Parkinson's is expected to fail
until its ZIP has been supplied and passes `check`.
""".strip())


def train_model(name, extra):
    config = DISEASES[name]
    if not dataset_check(name, config):
        raise SystemExit(2)
    command = [sys.executable, "-m", config["module"], *extra]
    print("Starting:", " ".join(command))
    raise SystemExit(subprocess.run(command, cwd=BACKEND).returncode)


def show_metrics(name):
    path = DISEASES[name]["metrics"]
    if not path.exists():
        raise SystemExit(f"Metrics do not exist yet. Train {name} first.")
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("guide", help="Print the recommended workflow")
    for command in ("status", "check"):
        child = subparsers.add_parser(command, help=f"{command.title()} datasets and models")
        child.add_argument("disease", choices=[*DISEASES, "all"], default="all", nargs="?")
    inspector = subparsers.add_parser("inspect", help="Show dataset files and CSV columns without training")
    inspector.add_argument("disease", choices=[*DISEASES, "all"], default="all", nargs="?")
    trainer = subparsers.add_parser("train", help="Train and evaluate one selected model")
    trainer.add_argument("disease", choices=DISEASES)
    trainer.add_argument("training_options", nargs=argparse.REMAINDER, help="Options passed to that model's trainer")
    metrics = subparsers.add_parser("metrics", help="Print saved evaluation metrics")
    metrics.add_argument("disease", choices=DISEASES)
    subparsers.add_parser("test", help="Run backend integration tests without MQTT")
    subparsers.add_parser("serve", help="Start MQTT real-time inference")

    args = parser.parse_args()
    if args.command in (None, "guide"):
        show_guide()
    elif args.command == "status":
        show_status(args.disease)
    elif args.command == "check":
        success = all(dataset_check(name, config) for name, config in selected(args.disease))
        raise SystemExit(0 if success else 2)
    elif args.command == "inspect":
        success = all(inspect_dataset(name, config) for name, config in selected(args.disease))
        raise SystemExit(0 if success else 2)
    elif args.command == "train":
        train_model(args.disease, args.training_options)
    elif args.command == "metrics":
        show_metrics(args.disease)
    elif args.command == "test":
        raise SystemExit(subprocess.run([sys.executable, "-m", "unittest", "-v", "test_backend.py"], cwd=BACKEND).returncode)
    elif args.command == "serve":
        raise SystemExit(subprocess.run([sys.executable, "main.py"], cwd=BACKEND).returncode)


if __name__ == "__main__":
    main()
