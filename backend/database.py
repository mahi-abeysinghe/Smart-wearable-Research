"""Portable SQLite storage for Smart Wear telemetry and diagnostic results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sqlite3
import threading
import time


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "smart_wear.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at REAL NOT NULL,
    device_id TEXT NOT NULL DEFAULT 'unknown',
    sensor_timestamp REAL,
    heart_rate REAL NOT NULL DEFAULT 0,
    spo2 REAL NOT NULL DEFAULT 0,
    temperature REAL NOT NULL DEFAULT 0,
    ecg_sample_count INTEGER NOT NULL DEFAULT 0,
    gps_valid INTEGER NOT NULL DEFAULT 0,
    latitude REAL,
    longitude REAL,
    diagnosis TEXT NOT NULL,
    severity INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    risk_score INTEGER NOT NULL DEFAULT 0,
    raw_payload_json TEXT NOT NULL,
    enriched_payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    diagnosis TEXT,
    severity INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    details_json TEXT NOT NULL,
    FOREIGN KEY (record_id) REFERENCES telemetry_records(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rule_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id INTEGER NOT NULL,
    diagnosis TEXT NOT NULL,
    severity INTEGER NOT NULL DEFAULT 0,
    measured_value REAL,
    rule_text TEXT NOT NULL,
    FOREIGN KEY (record_id) REFERENCES telemetry_records(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_telemetry_device_time
ON telemetry_records(device_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_diagnosis
ON telemetry_records(diagnosis, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_record ON ai_predictions(record_id);
CREATE INDEX IF NOT EXISTS idx_rule_record ON rule_alerts(record_id);
"""


class SmartWearDatabase:
    """Small SQLite repository that can be copied with the project."""

    def __init__(self, path=DEFAULT_DATABASE_PATH):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = DELETE")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    @staticmethod
    def _number(value, default=0.0):
        return float(value) if isinstance(value, (int, float)) else default

    def save_record(self, raw_payload, enriched_payload):
        models = enriched_payload.get("models") or {}
        ecg_samples = raw_payload.get("ecg_samples", raw_payload.get("ecg_raw", []))
        if not isinstance(ecg_samples, list):
            ecg_samples = [ecg_samples] if isinstance(ecg_samples, (int, float)) else []
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO telemetry_records (
                    received_at, device_id, sensor_timestamp, heart_rate, spo2,
                    temperature, ecg_sample_count, gps_valid, latitude, longitude,
                    diagnosis, severity, confidence, risk_score,
                    raw_payload_json, enriched_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(), str(raw_payload.get("device_id", "unknown")),
                    self._number(raw_payload.get("timestamp"), None),
                    self._number(raw_payload.get("heart_rate")), self._number(raw_payload.get("spo2")),
                    self._number(raw_payload.get("temperature")), len(ecg_samples),
                    int(bool(raw_payload.get("gps_valid"))), self._number(raw_payload.get("latitude"), None),
                    self._number(raw_payload.get("longitude"), None), str(enriched_payload.get("diagnosis", "UNKNOWN")),
                    int(enriched_payload.get("severity", 0)), self._number(enriched_payload.get("confidence")),
                    int(enriched_payload.get("risk_score", 0)),
                    json.dumps(raw_payload, separators=(",", ":"), ensure_ascii=False),
                    json.dumps(enriched_payload, separators=(",", ":"), ensure_ascii=False),
                ),
            )
            record_id = cursor.lastrowid
            for model_name in ("arrhythmia", "wesad_stress"):
                result = models.get(model_name)
                if not isinstance(result, dict):
                    continue
                self._connection.execute(
                    """INSERT INTO ai_predictions
                    (record_id, model_name, diagnosis, severity, confidence, status, details_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record_id, model_name, result.get("diagnosis"), int(result.get("severity", 0)),
                        self._number(result.get("confidence")), str(result.get("status", "unknown")),
                        json.dumps(result, separators=(",", ":"), ensure_ascii=False),
                    ),
                )
            rules = models.get("rules") or {}
            for alert in rules.get("alerts", []):
                self._connection.execute(
                    """INSERT INTO rule_alerts
                    (record_id, diagnosis, severity, measured_value, rule_text)
                    VALUES (?, ?, ?, ?, ?)""",
                    (
                        record_id, str(alert.get("diagnosis", "UNKNOWN")), int(alert.get("severity", 0)),
                        self._number(alert.get("value"), None), str(alert.get("rule", "")),
                    ),
                )
        return record_id

    def recent_records(self, limit=100):
        limit = max(1, min(int(limit), 10000))
        with self._lock:
            rows = self._connection.execute(
                """SELECT id, received_at, device_id, sensor_timestamp, heart_rate, spo2,
                temperature, ecg_sample_count, gps_valid, latitude, longitude,
                diagnosis, severity, confidence, risk_score
                FROM telemetry_records ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def counts(self):
        with self._lock:
            return {
                "telemetry_records": self._connection.execute("SELECT COUNT(*) FROM telemetry_records").fetchone()[0],
                "ai_predictions": self._connection.execute("SELECT COUNT(*) FROM ai_predictions").fetchone()[0],
                "rule_alerts": self._connection.execute("SELECT COUNT(*) FROM rule_alerts").fetchone()[0],
            }

    def close(self):
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def export_csv(database, destination, limit=10000):
    rows = database.recent_records(limit)
    destination = Path(destination)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Inspect or export the portable Smart Wear SQLite database")
    parser.add_argument("--database", default=str(DEFAULT_DATABASE_PATH))
    parser.add_argument("--recent", type=int, default=0, help="Print the newest N records as JSON")
    parser.add_argument("--export-csv", help="Export records to this CSV file")
    args = parser.parse_args()
    database = SmartWearDatabase(args.database)
    try:
        print(json.dumps(database.counts(), indent=2))
        if args.recent:
            print(json.dumps(database.recent_records(args.recent), indent=2))
        if args.export_csv:
            print(f"Exported {export_csv(database, args.export_csv)} records to {args.export_csv}")
    finally:
        database.close()


if __name__ == "__main__":
    main()
