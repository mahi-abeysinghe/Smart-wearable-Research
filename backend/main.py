"""Smart Wear MQTT inference backend.

ESP32 raw packets arrive on smartwear/device/telemetry. This service adds model
results and publishes one enriched packet to smartwear/device/predictions.
"""

from __future__ import annotations

from collections import deque
import atexit
import json
import os
import time

import paho.mqtt.client as mqtt

from AI_models.arrhythmia.inference import ArrhythmiaInference
from AI_models.wesad.inference import WesadInference
from database import DEFAULT_DATABASE_PATH, SmartWearDatabase


MQTT_BROKER = os.getenv("MQTT_BROKER", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC_TELEMETRY = os.getenv("MQTT_TELEMETRY_TOPIC", "smartwear/device/telemetry")
TOPIC_PREDICTIONS = os.getenv("MQTT_PREDICTIONS_TOPIC", "smartwear/device/predictions")
DATABASE_PATH = os.getenv("SMARTWEAR_DB_PATH", str(DEFAULT_DATABASE_PATH))


def new_history():
    return {
        "timestamps": deque(maxlen=3600),
        "ecg": deque(maxlen=2500),
        "heart_rate": deque(maxlen=600),
        "spo2": deque(maxlen=600),
        "temperature": deque(maxlen=600),
    }


class SmartWearBackend:
    def __init__(self, database_path=DATABASE_PATH):
        self.history = new_history()
        self.database = SmartWearDatabase(database_path)
        self.models = {
            "arrhythmia": ArrhythmiaInference(),
            "wesad_stress": WesadInference(),
        }

    def close(self):
        self.database.close()

    @staticmethod
    def _append(target, value):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (int, float)):
                    target.append(float(item))
        elif isinstance(value, (int, float)):
            target.append(float(value))

    def update_history(self, data):
        self.history["timestamps"].append(float(data.get("timestamp", time.time())))
        self._append(self.history["ecg"], data.get("ecg_samples", data.get("ecg_raw")))
        validators = {
            "heart_rate": lambda value: 20 <= value <= 250,
            "spo2": lambda value: 50 <= value <= 100,
            "temperature": lambda value: 20 <= value <= 45,
        }
        for field, validator in validators.items():
            value = data.get(field)
            if isinstance(value, (int, float)) and validator(float(value)):
                self.history[field].append(float(value))

    @staticmethod
    def evaluate_rules(data):
        alerts = []
        heart_rate = float(data.get("heart_rate", 0) or 0)
        spo2 = float(data.get("spo2", 0) or 0)
        temperature = float(data.get("temperature", 0) or 0)
        if 0 < spo2 < 85:
            alerts.append({"diagnosis": "SEVERE HYPOXEMIA", "severity": 2, "value": spo2, "rule": "SpO2 < 85%"})
        elif 0 < spo2 < 92:
            alerts.append({"diagnosis": "LOW OXYGEN SATURATION", "severity": 1, "value": spo2, "rule": "SpO2 < 92%"})
        if 0 < heart_rate < 50:
            alerts.append({"diagnosis": "BRADYCARDIA", "severity": 1, "value": heart_rate, "rule": "Heart rate < 50 bpm"})
        elif heart_rate > 140:
            alerts.append({"diagnosis": "SEVERE TACHYCARDIA", "severity": 2, "value": heart_rate, "rule": "Heart rate > 140 bpm"})
        elif heart_rate > 110:
            alerts.append({"diagnosis": "TACHYCARDIA", "severity": 1, "value": heart_rate, "rule": "Heart rate > 110 bpm"})
        if temperature >= 39.5:
            alerts.append({"diagnosis": "HIGH FEVER", "severity": 2, "value": temperature, "rule": "Temperature >= 39.5 C"})
        elif temperature >= 38.0:
            alerts.append({"diagnosis": "FEVER", "severity": 1, "value": temperature, "rule": "Temperature >= 38.0 C"})
        elif 0 < temperature < 35.0:
            alerts.append({"diagnosis": "LOW BODY TEMPERATURE", "severity": 2, "value": temperature, "rule": "Temperature < 35.0 C"})
        has_data = any(value > 0 for value in (heart_rate, spo2, temperature)) or bool(data.get("ecg_samples"))
        if not has_data:
            return {"diagnosis": "NO SENSOR DATA", "severity": 0, "confidence": 0.0, "status": "no_data", "alerts": []}
        if not alerts:
            return {"diagnosis": "SENSOR RULES NORMAL", "severity": 0, "confidence": 0.0, "status": "ready", "alerts": []}
        chosen = max(alerts, key=lambda alert: alert["severity"])
        return {**chosen, "confidence": 1.0, "status": "ready", "alerts": alerts}

    @staticmethod
    def decide(results):
        ready = [result for result in results.values() if result.get("status") == "ready"]
        if not ready:
            if any(result.get("status") == "no_data" for result in results.values()):
                return "NO SENSOR DATA", 0, 0.0
            return "ANALYSIS WARMING UP", 0, 0.0
        abnormal = [result for result in ready if result.get("severity", 0) > 0 and result.get("confidence", 0) >= .60]
        if not abnormal and not any(result.get("status") == "ready" for name, result in results.items() if name != "rules"):
            return "ANALYSIS WARMING UP", 0, 0.0
        if not abnormal:
            return "NORMAL HEALTH", 0, max((item.get("confidence", 0.0) for item in ready), default=0.0)
        chosen = max(abnormal, key=lambda item: (item.get("severity", 0), item.get("confidence", 0.0)))
        return chosen["diagnosis"], int(chosen["severity"]), float(chosen["confidence"])

    def process(self, payload):
        data = json.loads(payload) if isinstance(payload, (str, bytes, bytearray)) else dict(payload)
        if data.get("backend_processed"):
            raise ValueError("Packet has already been processed")
        self.update_history(data)
        results = {name: model.predict(data, self.history) for name, model in self.models.items()}
        results["rules"] = self.evaluate_rules(data)
        diagnosis, severity, confidence = self.decide(results)
        output = dict(data)
        output.update({
            "diagnosis": diagnosis,
            "severity": severity,
            "confidence": round(confidence, 6),
            "risk_score": 90 if severity >= 2 else (50 if severity == 1 else 0),
            "models": results,
            "backend_processed": True,
            "backend_timestamp": time.time(),
        })
        output["record_id"] = self.database.save_record(data, output)
        return output


backend = SmartWearBackend()
atexit.register(backend.close)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(TOPIC_TELEMETRY, qos=1)
    else:
        print(f"MQTT connection failed with code {rc}")


def on_message(client, userdata, message):
    try:
        output = backend.process(message.payload.decode("utf-8"))
        client.publish(TOPIC_PREDICTIONS, json.dumps(output, separators=(",", ":")), qos=1)
        print(f"{output.get('device_id', 'unknown')}: {output['diagnosis']} ({output['confidence']:.1%})")
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as error:
        print(f"Rejected telemetry packet: {error}")
    except Exception as error:
        print(f"Inference error: {error}")


def main():
    client = mqtt.Client(client_id=f"smartwear-backend-{os.getpid()}", clean_session=True)
    client.on_connect = on_connect
    client.on_message = on_message
    if os.getenv("MQTT_USERNAME"):
        client.username_pw_set(os.environ["MQTT_USERNAME"], os.getenv("MQTT_PASSWORD"))
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    print(f"Listening on {TOPIC_TELEMETRY}; publishing to {TOPIC_PREDICTIONS}")
    client.loop_forever()


if __name__ == "__main__":
    main()
