# Smart Wear backend and four-sensor implementation

This document describes the final live system for the ESP32-S3 Mini with these
four data-acquisition modules:

1. MAX30102 (or MAX30100 with a different Arduino driver): heart rate and SpO₂.
2. MLX90614: non-contact object temperature.
3. AD8232: single-lead ECG waveform.
4. NEO-7M: GPS position and satellite fix.

The SSD1306 is a display and does not generate model input.

> Research prototype only. AI outputs and threshold alerts are not medical
> diagnoses and are not validated for clinical or emergency use.

## 1. Final live scope

| Output | Source | Method | Live? |
|---|---|---|---|
| ECG beat class | AD8232 | Arrhythmia random forest | Yes |
| Experimental stress pattern | MAX30102 HR + MLX90614 temperature | Reduced WESAD random forest | Yes, experimental |
| Bradycardia/tachycardia | MAX30102 HR | Transparent thresholds | Yes |
| Low oxygen/severe hypoxemia | MAX30102 SpO₂ | Transparent thresholds | Yes |
| Fever/low temperature | MLX90614 | Transparent thresholds | Yes |
| Location | NEO-7M | GPS passthrough | Yes; not a disease signal |
| Parkinsonian motion | PADS wrist motion | Requires accelerometer/gyroscope | **No** |

The trained PADS model remains saved for offline research, but `main.py` does
not import it or use it in live risk because none of the four sensors measures
wrist acceleration or angular velocity. Inferring Parkinson's from ECG, pulse,
temperature, or GPS would be unsupported.

## 2. Data flow

```text
MAX30102 + MLX90614 + AD8232 + NEO-7M
  -> ESP32-S3 Mini
  -> MQTT smartwear/device/telemetry
  -> backend/main.py
       -> AD8232 ECG feature extraction -> arrhythmia model
       -> HR/temperature window -> reduced WESAD model
       -> HR/SpO2/temperature -> threshold rules
       -> result fusion
  -> MQTT smartwear/device/predictions
  -> dashboard/index.html
```

The dashboard renders only enriched prediction packets. Raw health telemetry is
consumed by Python first so unprocessed values are never presented as AI output.

## 3. Python components

```text
backend/
|-- main.py
|-- manage_models.py
|-- test_backend.py
|-- requirements.txt
|-- README.md
`-- AI_models/
    |-- arrhythmia/train.py
    |-- arrhythmia/inference.py
    |-- wesad/train.py
    |-- wesad/inference.py
    `-- Parkinsons/                 offline model only
```

The only Python environment is `backend/.venv`.

## 4. Arrhythmia training and live input

Training uses `MIT-BIH Arrhythmia Database.csv` from the supplied ECG ZIP:

- 100,689 labeled beats.
- Classes: normal (`N`), ventricular ectopic (`VEB`), supraventricular ectopic
  (`SVEB`), fusion (`F`), and unclassifiable (`Q`).
- 32 engineered two-lead features.
- Record-level 80/20 split prevents beats from one recording leaking across sets.
- Training normal beats are reduced to improve abnormal sensitivity.
- Median imputer plus a balanced 240-tree random forest.

Held-out result:

- Accuracy: 88.13%.
- Balanced accuracy: 49.01%.
- VEB recall: 81.44%.
- SVEB recall: 24.13%.
- Fusion recall: 0%.

The live AD8232 sends 250 samples each second at 250 Hz. Python waits for at
least 2.5 seconds, checks signal usability, identifies recent peaks, creates the
dataset feature schema, and predicts. The training data is two-lead while AD8232
is single-lead, so the live feature conversion is an approximation.

## 5. Sensor-compatible WESAD retraining

The original WESAD model used EDA and accelerometer features that the final
hardware does not provide. It was retrained using only features available from
the listed modules:

- Heart-rate mean, standard deviation, minimum, and maximum derived from WESAD
  wrist BVP (mapped to MAX30102 heart rate).
- Temperature mean, standard deviation, minimum, and maximum (mapped to MLX90614).
- 60-second windows with 30-second overlap.
- Labels: WESAD stress label 2 versus baseline/amusement labels 1 and 3.
- Subject-level split: 11 training subjects and 4 held-out subjects.
- Balanced 240-tree random forest.

New held-out result:

- Accuracy: **77.42%**.
- Balanced accuracy: **77.07%**.
- Stress recall: **76.19%**.
- Stress precision: **59.81%**.
- Confusion matrix: `[[152, 43], [20, 64]]`.

Live inference waits for 30 valid HR and temperature readings. Alert severity is
raised only at 75% model probability. This remains experimental because WESAD
used an Empatica wrist skin-temperature sensor, whereas MLX90614 often measures
a different body location and uses a different measurement method.

## 6. Transparent sensor rules

Rules are returned under `models.rules`, including the measured value and exact
trigger. Zero or missing readings do not trigger a rule.

| Rule ID | Trigger | Severity |
|---|---|---|
| `BRADYCARDIA` | 0 < HR < 50 bpm | Warning |
| `TACHYCARDIA` | HR > 110 bpm | Warning |
| `SEVERE TACHYCARDIA` | HR > 140 bpm | High |
| `LOW OXYGEN SATURATION` | 0 < SpO₂ < 92% | Warning |
| `SEVERE HYPOXEMIA` | 0 < SpO₂ < 85% | High |
| `FEVER` | Temperature ≥ 38.0 °C | Warning |
| `HIGH FEVER` | Temperature ≥ 39.5 °C | High |
| `LOW BODY TEMPERATURE` | 0 < temperature < 35.0 °C | High |

These thresholds are demonstrations. Sensor placement, motion, perfusion,
ambient temperature, and calibration can produce false readings.

## 7. ESP32 firmware

The sketch is `esp32_code/esp32_sensors.ino`. It targets MAX30102 with the
SparkFun MAX3010x library. A MAX30100 needs its own library and code adapter.

Required Arduino libraries:

- PubSubClient
- SparkFun MAX3010x Pulse and Proximity Sensor Library
- Adafruit MLX90614 Library
- TinyGPSPlus

Default example pins for ESP32-S3 Mini:

```text
AD8232 OUT     GPIO 4
AD8232 LO+     GPIO 5
AD8232 LO-     GPIO 6
I2C SDA        GPIO 8
I2C SCL        GPIO 9
NEO-7M TX      GPIO 16 (ESP RX)
NEO-7M RX      GPIO 17 (ESP TX)
```

Verify these pins against the exact ESP32-S3 Mini board before wiring.

The firmware publishes:

```json
{
  "device_id": "ESP32S3_WEAR_01",
  "heart_rate": 78,
  "spo2": 97,
  "temperature": 36.8,
  "ecg_sampling_rate": 250,
  "ecg_leads_off": false,
  "ecg_samples": [2048, 2060, 2082],
  "gps_valid": true,
  "latitude": 6.927100,
  "longitude": 79.861200,
  "gps_satellites": 7
}
```

The ECG array is shortened above for readability. Invalid sensor readings are
published as zero, allowing the webpage to distinguish no reading from a normal
reading.

## 8. Backend processing and output

`main.py` validates physiological ranges before adding values to history:

- HR: 20–250 bpm.
- SpO₂: 50–100%.
- Temperature: 20–45 °C.

Each prediction packet includes the raw fields plus:

```json
{
  "diagnosis": "NORMAL HEALTH",
  "severity": 0,
  "confidence": 0.44,
  "risk_score": 0,
  "models": {
    "arrhythmia": {"status": "ready", "diagnosis": "NORMAL HEALTH"},
    "wesad_stress": {"status": "ready", "diagnosis": "NO STRESS PATTERN"},
    "rules": {"status": "ready", "alerts": []}
  },
  "backend_processed": true
}
```

PADS/Parkinson's is intentionally absent from live `models`.

## 9. Dashboard behavior

The Diagnostics page is divided into two sections:

1. **Trained AI Model Diagnostics** at the top: arrhythmia classes trained from
   AD8232 ECG and the experimental WESAD HR/temperature stress probability.
2. **Rule-Based Sensor Diagnostics** underneath: transparent heart-rate, SpO₂,
   and temperature thresholds.

Unsupported GSR, posture, fall, motion, and Parkinson's risks are not rendered.
GPS is shown as fix/no-fix and coordinates; it is not assigned a health risk.

When no enriched packet arrives for five seconds, the dashboard resets:

- All numeric values to zero.
- All gauges and probability histories to zero.
- GPS fix to zero.
- Diagnosis to `NO SENSOR DATA`.
- Report values to zero.
- Alert styling and risk score to zero.

This prevents old readings from looking live after the ESP32 or broker stops.

## 10. Portable SQLite record database

The backend uses Python's built-in SQLite support, so no database server or
extra package is required. The portable database file is:

```text
backend/smart_wear.db
```

Every processed MQTT packet is stored in three related tables:

- `telemetry_records`: sensor values, GPS, final diagnosis, risk score, and the
  complete raw and enriched JSON packets.
- `ai_predictions`: separate arrhythmia and WESAD result rows for each packet.
- `rule_alerts`: one row for every threshold rule triggered by the packet.

The published MQTT prediction includes `record_id`, which identifies its saved
database record. The schema and storage code are in `backend/database.py`.

Inspect record counts and the newest records:

```powershell
.venv\Scripts\python.exe database.py --recent 10
```

Export up to 10,000 recent records to a normal CSV file:

```powershell
.venv\Scripts\python.exe database.py --export-csv smart_wear_records.csv
```

To place the database somewhere else, set `SMARTWEAR_DB_PATH` before starting
`main.py`. Stop the backend before copying the database file so the copy contains
the latest committed records.

## 11. MQTT setup

Mosquitto needs TCP and WebSocket listeners:

```text
listener 1883 0.0.0.0
allow_anonymous true

listener 9001 0.0.0.0
protocol websockets
```

Anonymous MQTT is for isolated local development only. Use authentication, ACLs,
and TLS outside a controlled laboratory network.

Set the ESP32 `mqtt_server` to the laptop's LAN IP. The backend defaults to
`127.0.0.1:1883`; the browser defaults to WebSockets at `127.0.0.1:9001`.

Start the backend:

```powershell
cd D:\SLIIT\projectr\smart_wear\smart_wear\backend
.venv\Scripts\python.exe main.py
```

Then open `dashboard/index.html` on the same laptop.

## 12. Training, metrics, and tests

```powershell
.venv\Scripts\python.exe manage_models.py train arrhythmia
.venv\Scripts\python.exe manage_models.py train wesad
.venv\Scripts\python.exe manage_models.py metrics arrhythmia
.venv\Scripts\python.exe manage_models.py metrics wesad
.venv\Scripts\python.exe manage_models.py test
```

PADS can still be trained for offline research, but doing so does not reconnect
it to `main.py`:

```powershell
.venv\Scripts\python.exe manage_models.py train parkinsons --sampling-rate 50
```

## 13. Verification completed

- WESAD was retrained with only HR and temperature features.
- Arrhythmia and reduced WESAD bundles load through `main.py`.
- Parkinson's is absent from live inference.
- Rule tests cover simultaneous severe HR, SpO₂, and temperature alerts.
- Eight backend tests pass, including SQLite telemetry/model/rule persistence.
- A 35-packet four-sensor simulation reaches ready AI inference.
- A missing-data packet returns `NO SENSOR DATA`.
- Dashboard JavaScript passes syntax validation.

The remaining physical verification is to install/configure Mosquitto, install
the Arduino libraries, confirm board pins, flash the ESP32-S3 Mini, and test with
the actual four modules.
