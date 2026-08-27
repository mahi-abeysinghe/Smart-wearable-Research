import math
from pathlib import Path
import tempfile
import unittest

from AI_models.arrhythmia.inference import ArrhythmiaInference, extract_realtime_features
from AI_models.wesad.inference import WesadInference
from database import SmartWearDatabase
from main import SmartWearBackend, new_history


class BackendTests(unittest.TestCase):
    def test_arrhythmia_model_exists_and_loads(self):
        self.assertIsNotNone(ArrhythmiaInference().bundle)

    def test_wesad_model_exists_and_loads(self):
        self.assertIsNotNone(WesadInference().bundle)

    def test_feature_extraction_has_model_schema(self):
        inference = ArrhythmiaInference()
        samples = [2048 + 500 * math.sin(i * 2 * math.pi / 200) for i in range(750)]
        row = extract_realtime_features(samples, inference.bundle["feature_names"], 250)
        self.assertIsNotNone(row)
        self.assertEqual(list(row.columns), inference.bundle["feature_names"])

    def test_warmup_is_not_reported_as_normal(self):
        backend = SmartWearBackend(":memory:")
        self.addCleanup(backend.close)
        output = backend.process({"device_id": "TEST", "ecg_samples": [1, 2, 3], "ecg_sampling_rate": 250})
        self.assertEqual(output["diagnosis"], "ANALYSIS WARMING UP")
        self.assertTrue(output["backend_processed"])

    def test_no_sensor_data_is_explicit(self):
        backend = SmartWearBackend(":memory:")
        self.addCleanup(backend.close)
        output = backend.process({"device_id": "TEST"})
        self.assertEqual(output["diagnosis"], "NO SENSOR DATA")

    def test_sensor_rules(self):
        result = SmartWearBackend.evaluate_rules({"heart_rate": 145, "spo2": 83, "temperature": 39.7})
        self.assertEqual(result["severity"], 2)
        self.assertEqual(len(result["alerts"]), 3)

    def test_leads_off_is_exposed(self):
        inference = ArrhythmiaInference()
        result = inference.predict({"ecg_leads_off": True}, new_history())
        self.assertEqual(result["status"], "poor_signal")

    def test_sqlite_database_keeps_telemetry_ai_and_rule_records(self):
        with tempfile.TemporaryDirectory() as folder:
            database_path = Path(folder) / "portable.db"
            backend = SmartWearBackend(database_path)
            try:
                output = backend.process({
                    "device_id": "TEST-ESP32", "heart_rate": 145,
                    "spo2": 83, "temperature": 39.7, "gps_valid": True,
                    "latitude": 6.9271, "longitude": 79.8612,
                })
                self.assertGreater(output["record_id"], 0)
                self.assertTrue(database_path.exists())
                self.assertEqual(backend.database.counts()["telemetry_records"], 1)
                self.assertEqual(backend.database.counts()["ai_predictions"], 2)
                self.assertEqual(backend.database.counts()["rule_alerts"], 3)
            finally:
                backend.close()


if __name__ == "__main__":
    unittest.main()
