"""
Flask Local Server for SmartGlove & Waist Band Dual-Node Platform.

Runs locally on the user's PC on the same Wi-Fi / Local Area Network (LAN)
as the ESP32 Glove (Wrist Node) and ESP32 Waist Band (Waist Node).

Features:
- Ingests direct HTTP POST telemetry from ESP32 Glove and Waist Node
- Runs AI inference (Arrhythmia, Parkinson's, Hypoxemia, Sepsis, Stress)
- Evaluates Clinical Rules, Multi-Sensor Correlation, Dual-Node Fusion,
  Early Sepsis Detection, Silent Hypoxia Decline, and Silent SOS Gestures
- Exposes REST API and Server-Sent Events (SSE) for real-time dashboards
- Optional background Firebase Realtime Database bridge synchronization
"""

import argparse
import json
import os
import queue
import socket
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, render_template_string, request

from main import SmartWearBackend
from simulator import DiseaseSimulator, DISEASES

app = Flask(__name__)

# Global instances
backend: Optional[SmartWearBackend] = None
sse_listeners: List[queue.Queue] = []
sse_lock = threading.Lock()
firebase_bridge: Optional[Any] = None
firebase_sync_queue: queue.Queue = queue.Queue(maxsize=1000)
firebase_worker_thread: Optional[threading.Thread] = None

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
DEFAULT_SETTINGS: Dict[str, Any] = {
    "sync": True,
    "firebase_enabled": True,
    "firebase_interval_sec": 1.0,
}
settings: Dict[str, Any] = dict(DEFAULT_SETTINGS)
settings_lock = threading.Lock()
_firebase_last_sync: Dict[str, float] = {}


def _process_simulation_packet(packet: Dict[str, Any]):
    """Inject clinical packet into the backend pipeline and dashboard."""
    if backend is None:
        return
    try:
        processed = backend.process(packet)
        processed["raw_packet"] = packet
        broadcast_event("telemetry", processed)
        if processed.get("emergency"):
            broadcast_event("emergency", {
                "device_id": processed.get("device_id"),
                "risk_score": processed.get("risk_score"),
                "alert": processed.get("alert"),
                "timestamp": time.time(),
            })
        enqueue_firebase(processed)
    except Exception:
        pass


def _reset_simulation_history():
    """Flush history buffers before a simulation so prior readings don't dilute test signals."""
    if backend is None:
        return
    for tid in ("HEALTH_GLOVE_01", "SMARTWEAR_01"):
        if tid in backend.histories:
            for k in ("ecg", "ax", "ay", "az", "gx", "gy", "gz", "heart_rate", "spo2", "temperature", "gsr_raw"):
                backend.histories[tid][k].clear()
        if tid in backend.last_screenings:
            backend.last_screenings[tid].clear()
    if hasattr(backend, "silent_hypoxia_detector"):
        backend.silent_hypoxia_detector._state.clear()
    if hasattr(backend, "early_sepsis_detector"):
        backend.early_sepsis_detector._state.clear()


def _on_simulation_status_change(status: Dict[str, Any]):
    broadcast_event("simulation_state", status)


simulator: DiseaseSimulator = DiseaseSimulator(
    process_fn=_process_simulation_packet,
    reset_fn=_reset_simulation_history,
    on_status_change=_on_simulation_status_change,
)


def load_settings():
    """Load persisted UI settings from settings.json (if present)."""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        with settings_lock:
            if isinstance(saved.get("sync"), bool):
                settings["sync"] = saved["sync"]
            if isinstance(saved.get("firebase_enabled"), bool):
                settings["firebase_enabled"] = saved["firebase_enabled"]
            if isinstance(saved.get("firebase_interval_sec"), (int, float)):
                settings["firebase_interval_sec"] = max(0.0, float(saved["firebase_interval_sec"]))
    except Exception:
        pass


def save_settings():
    try:
        with settings_lock:
            data = dict(settings)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def enqueue_firebase(processed: Dict[str, Any]):
    """Queue a processed packet for Firebase, honoring the on/off switch and interval."""
    if not firebase_bridge:
        return
    with settings_lock:
        enabled = settings["firebase_enabled"]
        interval = settings["firebase_interval_sec"]
    if not enabled:
        return
    key = str(processed.get("device_id", "unknown"))
    now = time.time()
    if not processed.get("emergency") and now - _firebase_last_sync.get(key, 0.0) < interval:
        return
    _firebase_last_sync[key] = now
    try:
        firebase_sync_queue.put_nowait(processed)
    except queue.Full:
        pass


def _firebase_sync_worker():
    """Background worker thread that pushes locally processed packets to Firebase."""
    while True:
        try:
            item = firebase_sync_queue.get()
            if item is None:
                break
            if firebase_bridge:
                firebase_bridge.sync_processed(item)
        except Exception:
            pass
        finally:
            firebase_sync_queue.task_done()



def get_local_ip_addresses() -> List[str]:
    """Auto-detect IPv4 addresses on all local network adapters."""
    ips = set()
    try:
        # Standard hostname resolution
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass

    try:
        # UDP connect probe (doesn't send actual packet, finds routing interface)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass

    return sorted(list(ips)) or ["127.0.0.1"]


def broadcast_event(event_type: str, data: Dict[str, Any]):
    """Push an SSE event to all connected dashboard clients."""
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with sse_lock:
        stale_listeners = []
        for q in sse_listeners:
            try:
                q.put_nowait(payload)
            except queue.Full:
                stale_listeners.append(q)
        for q in stale_listeners:
            if q in sse_listeners:
                sse_listeners.remove(q)


@app.after_request
def add_cors_headers(response):
    """Enable cross-origin requests for local web dashboards."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ==============================================================================
#  HTTP Ingestion Endpoints (For ESP32 Nodes)
# ==============================================================================

@app.route("/api/telemetry", methods=["POST"])
def ingest_telemetry():
    """Primary telemetry ingest endpoint for ESP32 Glove and Waist Node.

    Expects JSON payload with sensor readings:
    - Glove: { "device_id": "HEALTH_GLOVE_01", "heart_rate": 78, "spo2": 98,
               "temperature": 36.6, "gsr_raw": 1200, "ecg_samples": [...], ... }
    - Waist: { "device_id": "WAIST_NODE", "node_type": "waist",
               "ax": 0.1, "ay": 0.2, "az": 9.8, "tilt_degrees": 5.0, ... }
    """
    if backend is None:
        return jsonify({"error": "Backend not initialized"}), 500

    raw_data = request.get_json(silent=True)
    if not raw_data or not isinstance(raw_data, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    # If disease simulation is running, ignore physical glove data
    if simulator and simulator.active and not raw_data.get("simulated"):
        node_type = str(raw_data.get("node_type", "")).lower()
        dev_id = str(raw_data.get("device_id", "")).upper()
        if node_type == "glove" or "GLOVE" in dev_id or (node_type != "waist" and "WAIST" not in dev_id):
            return jsonify({
                "status": "ignored",
                "message": "Physical glove telemetry ignored while disease simulation is active.",
                "simulation": simulator.status(),
            }), 200

    try:
        processed = backend.process(raw_data)
    except Exception as exc:
        return jsonify({"error": f"Processing error: {str(exc)}"}), 500

    # Attach the exact raw telemetry packet for frontend inspection
    processed["raw_packet"] = raw_data

    # Broadcast to live dashboard clients via SSE
    broadcast_event("telemetry", processed)
    if processed.get("emergency"):
        broadcast_event("emergency", {
            "device_id": processed.get("device_id"),
            "risk_score": processed.get("risk_score"),
            "alert": processed.get("alert"),
            "timestamp": time.time(),
        })

    # Asynchronously push to Firebase for mobile/web app
    enqueue_firebase(processed)

    return jsonify({
        "status": "ok",
        "sync": bool(settings.get("sync", False)),
        "device_id": processed.get("device_id"),
        "node_type": processed.get("node_type"),
        "risk_score": processed.get("risk_score"),
        "emergency": processed.get("emergency", False),
        "alert": processed.get("alert"),
        "screenings": processed.get("screenings"),
        "correlation": processed.get("correlation"),
        "sensor_fusion": processed.get("sensor_fusion"),
        "early_sepsis": processed.get("early_sepsis"),
        "silent_hypoxia": processed.get("silent_hypoxia"),
        "sos_gesture": processed.get("sos_gesture"),
    }), 200


@app.route("/api/telemetry/glove", methods=["POST"])
def ingest_glove():
    """Convenience endpoint specifically tagged as glove node."""
    data = request.get_json(silent=True) or {}
    data.setdefault("node_type", "glove")
    data.setdefault("device_id", "HEALTH_GLOVE_01")
    return _process_and_respond(data)


@app.route("/api/telemetry/waist", methods=["POST"])
def ingest_waist():
    """Convenience endpoint specifically tagged as waist node."""
    data = request.get_json(silent=True) or {}
    data.setdefault("node_type", "waist")
    data.setdefault("device_id", "WAIST_NODE")
    return _process_and_respond(data)


def _process_and_respond(data: Dict[str, Any]):
    if simulator and simulator.active and not data.get("simulated"):
        node_type = str(data.get("node_type", "")).lower()
        dev_id = str(data.get("device_id", "")).upper()
        if node_type == "glove" or "GLOVE" in dev_id:
            return jsonify({
                "status": "ignored",
                "message": "Physical glove telemetry ignored while disease simulation is active.",
                "simulation": simulator.status(),
            }), 200
    if backend is None:
        return jsonify({"error": "Backend not initialized"}), 500
    try:
        processed = backend.process(data)
        processed["raw_packet"] = data
        processed["sync"] = bool(settings.get("sync", False))
        broadcast_event("telemetry", processed)
        enqueue_firebase(processed)
        return jsonify(processed), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ==============================================================================
#  Dashboard & Query REST Endpoints
# ==============================================================================

@app.route("/api/simulate/status", methods=["GET"])
def api_simulate_status():
    """Get active disease simulation status."""
    if not simulator:
        return jsonify({"active": False, "diseases": []})
    return jsonify(simulator.status())


@app.route("/api/simulate/start", methods=["POST"])
def api_simulate_start():
    """Start clinical disease simulation, ignoring incoming glove data."""
    if not simulator:
        return jsonify({"error": "Simulator not initialized"}), 500
    body = request.get_json(silent=True) or {}
    disease = body.get("disease")
    if not disease:
        return jsonify({"error": "Missing disease parameter"}), 400
    try:
        status = simulator.start(disease)
        return jsonify(status), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/simulate/stop", methods=["POST"])
def api_simulate_stop():
    """Stop active simulation and return to live physical glove telemetry."""
    if not simulator:
        return jsonify({"active": False}), 200
    simulator.stop()
    return jsonify(simulator.status()), 200

@app.route("/api/settings", methods=["GET", "POST", "OPTIONS"])
def api_settings():
    """Read or update runtime settings (Firebase sync on/off and interval)."""
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        sync_changed = False
        with settings_lock:
            if "sync" in body:
                val = bool(body["sync"])
                if settings.get("sync") != val:
                    settings["sync"] = val
                    sync_changed = True
            if "firebase_enabled" in body:
                settings["firebase_enabled"] = bool(body["firebase_enabled"])
            if "firebase_interval_sec" in body:
                try:
                    val = float(body["firebase_interval_sec"])
                    settings["firebase_interval_sec"] = min(max(val, 0.0), 3600.0)
                except (TypeError, ValueError):
                    return jsonify({"error": "firebase_interval_sec must be a number"}), 400
        save_settings()
        if sync_changed:
            broadcast_event("settings_updated", {"sync": settings.get("sync", False)})
    with settings_lock:
        out = dict(settings)
    out["firebase_available"] = firebase_bridge is not None
    return jsonify(out)


@app.route("/api/settings/sync", methods=["GET", "POST"])
def api_settings_sync():
    """Quick GET/POST endpoint to read or toggle background sync."""
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        with settings_lock:
            if "sync" in body:
                settings["sync"] = bool(body["sync"])
            else:
                settings["sync"] = not settings.get("sync", False)
            current_sync = settings["sync"]
        save_settings()
        broadcast_event("settings_updated", {"sync": current_sync})
        return jsonify({"status": "ok", "sync": current_sync})
    with settings_lock:
        return jsonify({"sync": bool(settings.get("sync", False))})


@app.route("/api/status", methods=["GET"])
def get_status():
    """Server health check and connected network information."""
    local_ips = get_local_ip_addresses()
    db_counts = backend.database.counts() if backend else {}
    return jsonify({
        "status": "online",
        "service": "SmartGlove AI Healthcare Hub",
        "server_time": time.time(),
        "local_ip_addresses": local_ips,
        "database": db_counts,
        "active_sse_clients": len(sse_listeners),
    })


@app.route("/api/latest", methods=["GET"])
def get_latest_all():
    """Retrieve the latest telemetry snapshot for all devices."""
    if backend is None:
        return jsonify({"error": "Backend not initialized"}), 500

    admin_user = {"id": 1, "role": "admin"}
    records = backend.database.latest_records_by_device(admin_user)
    return jsonify({
        "status": "ok",
        "count": len(records),
        "devices": records,
    })


@app.route("/api/latest/<device_id>", methods=["GET"])
def get_latest_device(device_id: str):
    """Retrieve the latest telemetry snapshot for a specific device."""
    if backend is None:
        return jsonify({"error": "Backend not initialized"}), 500

    rec = backend.database.latest_record(device_id)
    if not rec:
        return jsonify({"status": "not_found", "message": f"No telemetry for {device_id}"}), 404
    return jsonify({"status": "ok", "record": rec})


@app.route("/api/raw/latest", methods=["GET"])
def get_latest_raw():
    """Retrieve the latest raw telemetry packet received from hardware nodes."""
    if backend is None:
        return jsonify({"error": "Backend not initialized"}), 500

    records = backend.database.recent_raw_records(limit=1)
    if records:
        return jsonify({
            "status": "ok",
            "record_id": records[0]["record_id"],
            "raw_packet": records[0]["raw_packet"],
            "created_at": records[0]["created_at"],
        })
    return jsonify({"status": "empty", "message": "No telemetry packets received yet"}), 200


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Retrieve active and recent emergency alerts."""
    if backend is None:
        return jsonify({"error": "Backend not initialized"}), 500

    alerts = backend.database.active_alerts()
    return jsonify({
        "status": "ok",
        "active_alerts": alerts,
        "count": len(alerts),
    })


@app.route("/api/alerts/resolve", methods=["POST"])
def resolve_alert():
    """Resolve an active emergency alert."""
    if backend is None:
        return jsonify({"error": "Backend not initialized"}), 500

    data = request.get_json(silent=True) or {}
    alert_id = data.get("alert_id")
    device_id = data.get("device_id")

    if alert_id and str(alert_id).lower() != "all":
        try:
            resolved = backend.database.resolve_alert(int(alert_id))
        except (ValueError, TypeError):
            resolved = backend.database.resolve_active_alerts(device_id)
    else:
        resolved = backend.database.resolve_active_alerts(device_id)

    broadcast_event("alert_resolved", {"alert_id": alert_id, "resolved": resolved})
    return jsonify({"status": "ok", "resolved": resolved})


@app.route("/api/facial-stress", methods=["POST"])
def analyze_facial_stress():
    """Direct REST endpoint for YOLOv8 facial expression stress detection.

    Expects JSON:
      { "image_base64": "data:image/jpeg;base64,...", "device_id": "HEALTH_GLOVE_01" }
    or multipart form-data with file field 'image'.
    """
    if backend is None:
        return jsonify({"error": "Backend not initialized"}), 500

    image_data = None
    device_id = "HEALTH_GLOVE_01"

    if request.is_json:
        body = request.get_json(silent=True) or {}
        image_data = body.get("image_base64") or body.get("image")
        device_id = body.get("device_id", device_id)
    elif "image" in request.files:
        uploaded_file = request.files["image"]
        image_data = uploaded_file.read()
        device_id = request.form.get("device_id", device_id)

    if not image_data:
        return jsonify({"error": "No image provided (send 'image_base64' in JSON or 'image' file)"}), 400

    result = backend.facial_stress_detector.predict(image_data)
    result["device_id"] = device_id

    # Broadcast event to local SSE listeners
    broadcast_event("facial_stress", result)

    # If Firebase is active, sync result to Firebase /facial_stress/result
    if firebase_bridge:
        try:
            res_ref = firebase_bridge.get_reference("facial_stress/result")
            res_ref.set(result)
        except Exception as exc:
            pass

    return jsonify(result), 200


@app.route("/api/facial-stress/latest", methods=["GET"])
def get_latest_facial_stress():
    """Retrieve the latest facial expression stress prediction result."""
    if backend is None:
        return jsonify({"error": "Backend not initialized"}), 500

    if firebase_bridge:
        try:
            res = firebase_bridge.get_reference("facial_stress/result").get()
            if res:
                return jsonify({"status": "ok", "result": res})
        except Exception:
            pass

    return jsonify({
        "status": "waiting_for_input",
        "message": "No facial stress analysis performed yet. Post image to /api/facial-stress or set /facial_stress/request in Firebase.",
    })



# ==============================================================================
#  Real-Time SSE Stream for Browser Dashboards
# ==============================================================================

@app.route("/api/stream", methods=["GET"])
def sse_stream():
    """Server-Sent Events (SSE) stream for live browser dashboard updates."""
    client_queue = queue.Queue(maxsize=50)
    with sse_lock:
        sse_listeners.append(client_queue)

    def event_generator():
        # Send initial connection handshake
        yield f"event: connected\ndata: {json.dumps({'message': 'Connected to SmartGlove Realtime SSE Stream', 'time': time.time()})}\n\n"
        try:
            while True:
                try:
                    msg = client_queue.get(timeout=20.0)
                    yield msg
                except queue.Empty:
                    # Keep-alive heartbeat
                    yield f": keepalive {time.time()}\n\n"
        finally:
            with sse_lock:
                if client_queue in sse_listeners:
                    sse_listeners.remove(client_queue)

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ==============================================================================
#  Embedded Web Dashboard UI
# ==============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SmartGlove AI Healthcare Hub â€” Local Server</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --surface: rgba(18, 24, 38, 0.85);
      --border: rgba(255, 255, 255, 0.08);
      --accent-cyan: #06b6d4;
      --accent-purple: #8b5cf6;
      --accent-red: #ef4444;
      --accent-amber: #f59e0b;
      --accent-green: #10b981;
      --text: #f3f4f6;
      --text-dim: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      background-image: radial-gradient(circle at 10% 20%, rgba(6, 182, 212, 0.08) 0%, transparent 40%),
                        radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
      color: var(--text);
      font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
      min-height: 100vh;
      padding: 24px;
    }
    .container { max-width: 1280px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
    }
    .brand { display: flex; align-items: center; gap: 14px; }
    .brand-icon {
      width: 44px; height: 44px; border-radius: 12px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; font-size: 20px; color: #fff;
    }
    h1 { font-size: 24px; font-weight: 600; letter-spacing: -0.5px; }
    .badge-status {
      display: flex; align-items: center; gap: 8px;
      background: rgba(16, 185, 129, 0.12); color: var(--accent-green);
      padding: 6px 14px; border-radius: 999px; font-size: 13px; font-weight: 500;
      border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .pulse-dot {
      width: 8px; height: 8px; border-radius: 50%; background: var(--accent-green);
      box-shadow: 0 0 10px var(--accent-green); animation: pulse 1.8s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-bottom: 24px; }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      backdrop-filter: blur(12px);
    }
    .card-header {
      display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;
    }
    .card-title { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); font-weight: 600; }
    .vital-value { font-size: 32px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
    .vital-unit { font-size: 14px; color: var(--text-dim); font-weight: 400; margin-left: 4px; }

    .endpoints-box {
      background: rgba(0, 0, 0, 0.4);
      border-radius: 12px;
      padding: 14px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      color: #38bdf8;
      margin-top: 12px;
      word-break: break-all;
    }
    .emergency-banner {
      display: none;
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.05));
      border: 1px solid rgba(239, 68, 68, 0.4);
      border-radius: 16px;
      padding: 18px 24px;
      margin-bottom: 24px;
      align-items: center;
      justify-content: space-between;
    }
    .emergency-active { display: flex !important; animation: glowRed 2s infinite alternate; }
    @keyframes glowRed { from { box-shadow: 0 0 15px rgba(239, 68, 68, 0.2); } to { box-shadow: 0 0 30px rgba(239, 68, 68, 0.5); } }

    .screening-row {
      display: flex; justify-content: space-between; align-items: center;
      padding: 10px 0; border-bottom: 1px solid var(--border);
    }
    .screening-row:last-child { border-bottom: none; }
    .prob-bar-container { width: 100px; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; }
    .prob-bar { height: 100%; border-radius: 3px; background: var(--accent-cyan); width: 0%; transition: width 0.4s ease; }

    .raw-log {
      background: rgba(0, 0, 0, 0.5);
      border-radius: 12px;
      padding: 14px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: #a7f3d0;
      height: 180px;
      overflow-y: auto;
    }

    /* Top Hero Risk Card & Dual-Node Sensor Grid */
    .risk-hero-card {
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .dual-node-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }
    .sub-sensor-tile {
      background: rgba(0, 0, 0, 0.28);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      transition: border-color 0.2s ease;
    }
    .sub-sensor-tile:hover {
      border-color: rgba(255, 255, 255, 0.16);
    }
    @media (max-width: 960px) {
      .dual-node-grid {
        grid-template-columns: 1fr !important;
      }
      .risk-hero-content {
        grid-template-columns: 1fr !important;
        gap: 16px !important;
      }
      .risk-hero-content > div {
        border-right: none !important;
        border-left: none !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
      }
    }

    /* Glove Raw Telemetry Section Styles */
    .tabs-nav {
      display: flex; gap: 8px; border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 16px;
    }
    .tab-btn {
      background: rgba(255, 255, 255, 0.05); color: var(--text-dim); border: 1px solid var(--border);
      padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 500;
      transition: all 0.2s ease;
    }
    .tab-btn:hover { background: rgba(255, 255, 255, 0.1); color: var(--text); }
    .tab-btn.active {
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(139, 92, 246, 0.2));
      color: #38bdf8; border-color: rgba(6, 182, 212, 0.5); font-weight: 600;
    }
    .btn-action {
      background: rgba(255, 255, 255, 0.07); color: var(--text); border: 1px solid var(--border);
      padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-family: 'JetBrains Mono', monospace;
      display: inline-flex; align-items: center; gap: 6px; transition: all 0.2s ease;
    }
    .btn-action:hover { background: rgba(255, 255, 255, 0.14); border-color: rgba(255, 255, 255, 0.2); }
    .btn-action.paused {
      background: rgba(245, 158, 11, 0.2); color: #fbbf24; border-color: rgba(245, 158, 11, 0.4);
    }
    .raw-sensors-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px;
    }
    .sensor-tile {
      background: rgba(0, 0, 0, 0.35); border: 1px solid var(--border); border-radius: 12px; padding: 14px;
      display: flex; flex-direction: column; justify-content: space-between;
    }
    .sensor-tile-title {
      font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--accent-cyan); font-weight: 600; margin-bottom: 8px;
      display: flex; justify-content: space-between; align-items: center;
    }
    .sensor-tile-val {
      font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 600; color: #fff; margin-bottom: 4px;
    }
    .sensor-tile-sub {
      font-size: 12px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace;
    }
    .raw-json-viewer {
      background: #060911; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 16px;
      font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #7dd3fc; line-height: 1.5;
      max-height: 420px; overflow-y: auto; overflow-x: auto; white-space: pre-wrap; word-break: break-all;
    }
    .badge-pill {
      font-size: 11px; padding: 3px 8px; border-radius: 999px; font-weight: 600; font-family: 'JetBrains Mono', monospace;
    }
    .badge-pill.green { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-pill.amber { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-pill.cyan { background: rgba(6, 182, 212, 0.2); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); }

    /* Settings UI */
    .header-right { display: flex; align-items: center; gap: 12px; }
    .btn-settings {
      background: rgba(255,255,255,0.06); color: var(--text); border: 1px solid var(--border);
      padding: 7px 14px; border-radius: 999px; cursor: pointer; font-size: 13px; font-weight: 500;
      font-family: inherit; transition: all 0.2s ease;
    }
    .btn-settings:hover { background: rgba(6,182,212,0.15); border-color: rgba(6,182,212,0.5); color: #38bdf8; }
    .modal-overlay {
      position: fixed; inset: 0; background: rgba(5,8,15,0.7); backdrop-filter: blur(6px);
      display: none; align-items: center; justify-content: center; z-index: 1000;
    }
    .modal-overlay.open { display: flex; }
    .modal {
      width: min(440px, 92vw); background: #121826; border: 1px solid var(--border);
      border-radius: 18px; padding: 24px; box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    }
    .modal h2 { font-size: 18px; font-weight: 600; margin-bottom: 4px; }
    .modal .sub { font-size: 12px; color: var(--text-dim); margin-bottom: 18px; }
    .setting-row {
      display: flex; justify-content: space-between; align-items: center; gap: 16px;
      padding: 14px 0; border-top: 1px solid var(--border);
    }
    .setting-label { font-size: 14px; font-weight: 500; }
    .setting-desc { font-size: 12px; color: var(--text-dim); margin-top: 2px; }
    .switch { position: relative; width: 46px; height: 26px; flex-shrink: 0; }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute; inset: 0; background: rgba(255,255,255,0.15); border-radius: 999px;
      cursor: pointer; transition: 0.25s;
    }
    .slider::before {
      content: ''; position: absolute; width: 20px; height: 20px; left: 3px; top: 3px;
      background: #fff; border-radius: 50%; transition: 0.25s;
    }
    .switch input:checked + .slider { background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); }
    .switch input:checked + .slider::before { transform: translateX(20px); }
    .num-input {
      width: 90px; background: rgba(0,0,0,0.4); color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: 7px 10px; font-family: 'JetBrains Mono', monospace; font-size: 13px;
    }
    .num-input:focus { outline: none; border-color: var(--accent-cyan); }
    .modal-actions { display: flex; justify-content: flex-end; align-items: center; gap: 10px; margin-top: 18px; }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); color: #fff; border: none;
      padding: 8px 18px; border-radius: 8px; cursor: pointer; font-weight: 600; font-family: inherit;
    }
    .btn-ghost {
      background: transparent; color: var(--text-dim); border: 1px solid var(--border);
      padding: 8px 16px; border-radius: 8px; cursor: pointer; font-family: inherit;
    }
    #settings-msg { font-size: 12px; color: var(--accent-green); margin-right: auto; }

    /* Disease Simulation Panel */
    .sim-card {
      background: linear-gradient(135deg, rgba(18, 24, 38, 0.95), rgba(30, 27, 75, 0.55));
      border: 1px solid rgba(139, 92, 246, 0.35);
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
      backdrop-filter: blur(14px);
    }
    .sim-header {
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; margin-bottom: 16px;
    }
    .sim-title-group { display: flex; align-items: center; gap: 12px; }
    .sim-icon {
      width: 42px; height: 42px; border-radius: 10px;
      background: linear-gradient(135deg, var(--accent-purple), var(--accent-cyan));
      display: flex; align-items: center; justify-content: center; font-size: 20px;
      box-shadow: 0 0 15px rgba(139, 92, 246, 0.4);
    }
    .sim-heading { font-size: 16px; font-weight: 700; color: #fff; letter-spacing: -0.3px; }
    .sim-subheading { font-size: 12px; color: var(--text-dim); margin-top: 2px; }
    .sim-status-wrapper { display: flex; align-items: center; gap: 10px; }
    .sim-badge-idle {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 6px 14px; border-radius: 999px; font-size: 12px; font-weight: 600;
      background: rgba(16, 185, 129, 0.12); color: var(--accent-green);
      border: 1px solid rgba(16, 185, 129, 0.3); font-family: 'JetBrains Mono', monospace;
    }
    .sim-badge-active {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 6px 16px; border-radius: 999px; font-size: 12px; font-weight: 700;
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(245, 158, 11, 0.25));
      color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.6);
      animation: simBadgeGlow 1.4s infinite alternate; font-family: 'JetBrains Mono', monospace;
    }
    @keyframes simBadgeGlow {
      from { box-shadow: 0 0 8px rgba(239, 68, 68, 0.3); }
      to { box-shadow: 0 0 22px rgba(239, 68, 68, 0.7); }
    }
    .sim-dot-idle { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-green); }
    .sim-dot-pulse {
      width: 8px; height: 8px; border-radius: 50%; background: var(--accent-red);
      box-shadow: 0 0 10px var(--accent-red); animation: pulse 1.2s infinite;
    }
    .sim-btn-stop {
      background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.45);
      padding: 6px 14px; border-radius: 999px; font-size: 12px; font-weight: 600;
      cursor: pointer; transition: all 0.2s ease; font-family: inherit;
    }
    .sim-btn-stop:hover { background: rgba(239, 68, 68, 0.35); border-color: rgba(239, 68, 68, 0.7); }
    .sim-progress-container {
      background: rgba(0, 0, 0, 0.35); border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px; padding: 12px 14px; margin-bottom: 14px;
    }
    .sim-progress-info {
      display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;
      font-size: 12px; color: #a5f3fc;
    }
    .sim-progress-track {
      width: 100%; height: 6px; background: rgba(255, 255, 255, 0.1); border-radius: 3px; overflow: hidden;
    }
    .sim-progress-fill {
      height: 100%; border-radius: 3px;
      background: linear-gradient(90deg, #ec4899, #8b5cf6, #06b6d4);
      transition: width 0.3s ease;
    }
    .sim-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;
    }
    .sim-btn {
      background: rgba(255, 255, 255, 0.04); border: 1px solid var(--border);
      border-radius: 12px; padding: 14px; text-align: left; cursor: pointer;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); display: flex; flex-direction: column;
      gap: 6px; font-family: inherit; color: inherit;
    }
    .sim-btn:hover {
      background: rgba(255, 255, 255, 0.08); border-color: rgba(6, 182, 212, 0.5);
      transform: translateY(-2px); box-shadow: 0 6px 18px rgba(6, 182, 212, 0.15);
    }
    .sim-btn.simulating {
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(139, 92, 246, 0.2));
      border-color: rgba(239, 68, 68, 0.6); box-shadow: 0 0 20px rgba(239, 68, 68, 0.35);
      animation: simBtnPulse 1.6s infinite alternate;
    }
    @keyframes simBtnPulse {
      from { border-color: rgba(239, 68, 68, 0.5); }
      to { border-color: rgba(245, 158, 11, 0.8); }
    }
    .sim-btn-top { display: flex; justify-content: space-between; align-items: center; }
    .sim-btn-icon { font-size: 22px; }
    .sim-btn-tag {
      font-size: 10px; font-family: 'JetBrains Mono', monospace; font-weight: 600;
      padding: 2px 6px; border-radius: 4px; background: rgba(255, 255, 255, 0.08);
      color: var(--accent-cyan);
    }
    .sim-btn-name { font-size: 14px; font-weight: 700; color: #fff; margin-top: 2px; }
    .sim-btn-desc { font-size: 11px; color: var(--text-dim); line-height: 1.3; }

  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <div class="brand-icon">SG</div>
        <div>
          <h1>SmartGlove AI Healthcare Hub</h1>
          <p style="font-size: 13px; color: var(--text-dim);">Local Flask Ingestion Server & Live Telemetry Monitor</p>
        </div>
      </div>
      <div class="header-right">
        <button id="btn-open-settings" class="btn-settings" onclick="openSettings()">âš™ Settings</button>
        <div class="badge-status">
          <div class="pulse-dot"></div>
          <span id="conn-status">Connecting SSE...</span>
        </div>
      </div>
    </header>

    <div id="settings-modal" class="modal-overlay" onclick="if(event.target===this)closeSettings()">
      <div class="modal" role="dialog" aria-labelledby="settings-title">
        <h2 id="settings-title">Settings</h2>
        <p class="sub" id="fb-avail-note">Hardware node telemetry & cloud options</p>
        <div class="setting-row">
          <div>
            <div class="setting-label">Sync</div>
            <div class="setting-desc">Synchronize continuous vitals telemetry stream</div>
          </div>
          <label class="switch">
            <input type="checkbox" id="set-sync-enabled" onchange="saveSettings()">
            <span class="slider"></span>
          </label>
        </div>
        <div class="setting-row">
          <div>
            <div class="setting-label">Firebase updates</div>
            <div class="setting-desc">Push telemetry to Firebase for the mobile app</div>
          </div>
          <label class="switch">
            <input type="checkbox" id="set-fb-enabled">
            <span class="slider"></span>
          </label>
        </div>
        <div class="setting-row">
          <div>
            <div class="setting-label">Update interval</div>
            <div class="setting-desc">Seconds between Firebase updates (0 = every packet). Emergencies always sent.</div>
          </div>
          <input type="number" class="num-input" id="set-fb-interval" min="0" max="3600" step="0.5">
        </div>
        <div class="modal-actions">
          <span id="settings-msg"></span>
          <button class="btn-ghost" id="btn-settings-close" onclick="closeSettings()">Close</button>
          <button class="btn-primary" id="btn-settings-save" onclick="saveSettings()">Save</button>
        </div>
      </div>
    </div>

    <div id="emergency-banner" class="emergency-banner">
      <div>
        <h2 style="color: var(--accent-red); font-size: 18px; margin-bottom: 4px;">EMERGENCY ALERT TRIGGERED</h2>
        <p id="emergency-detail" style="font-size: 14px; color: #fca5a5;">High physiological risk score detected.</p>
      </div>
      <button onclick="dismissEmergency()" style="background: var(--accent-red); color:#fff; border:none; padding:8px 16px; border-radius:8px; cursor:pointer; font-weight:600;">Dismiss Alert</button>
    </div>

    <!-- 1. Risk Score & Emergency Card (On Top) -->
    <div class="card risk-hero-card" style="margin-bottom: 20px; border: 1px solid rgba(139, 92, 246, 0.35); background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(30, 27, 75, 0.4) 100%); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);">
      <div class="card-header" style="flex-wrap: wrap; gap: 12px; margin-bottom: 14px;">
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="width: 34px; height: 34px; border-radius: 10px; background: rgba(139, 92, 246, 0.2); display: flex; align-items: center; justify-content: center; font-size: 18px;">ðŸ›¡ï¸</div>
          <div>
            <span class="card-title" style="color: #c084fc; font-size: 14px; letter-spacing: 1.2px; font-weight: 700;">Risk Score & Emergency</span>
            <div style="font-size: 11px; color: var(--text-dim);">Real-Time Dual-Node Clinical Risk Synthesis</div>
          </div>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
          <span class="badge-pill" id="badge-risk-status" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3);">NORMAL VITALS</span>
          <span class="badge-pill cyan" id="val-posture">Posture: STANDING</span>
          <span class="badge-pill" id="badge-fall-status" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3);">NO FALLS</span>
        </div>
      </div>

      <div class="risk-hero-content" style="display: grid; grid-template-columns: auto 1fr auto; gap: 24px; align-items: center;">
        <!-- Left: Big Risk Score -->
        <div style="display: flex; align-items: baseline; gap: 6px; padding-right: 24px; border-right: 1px solid var(--border);">
          <span class="vital-value" id="val-risk" style="font-size: 46px; color: var(--accent-green); line-height: 1;">0</span>
          <span class="vital-unit" style="font-size: 18px; color: var(--text-dim);">/ 100</span>
        </div>

        <!-- Middle: Dynamic Risk Bar & Clinical Summary -->
        <div style="display: flex; flex-direction: column; gap: 8px;">
          <div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px;">
            <span id="risk-level-label" style="font-weight: 600; color: var(--accent-green); text-transform: uppercase; letter-spacing: 0.5px;">Low Physiological Risk (Stable)</span>
            <span id="risk-detail-text" style="color: var(--text-dim); font-size: 11px;">Cross-verified by Dual-Node Sensor Fusion</span>
          </div>
          <div style="height: 10px; width: 100%; background: rgba(255, 255, 255, 0.08); border-radius: 5px; overflow: hidden; position: relative;">
            <div id="risk-progress-bar" style="height: 100%; width: 0%; background: linear-gradient(90deg, #10b981, #06b6d4); border-radius: 5px; transition: width 0.4s ease, background 0.4s ease;"></div>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace;">
            <span>0 SAFE</span>
            <span>30 WARNING</span>
            <span>60 ELEVATED</span>
            <span>100 CRITICAL</span>
          </div>
        </div>

        <!-- Right: Node Sync Status Pills -->
        <div style="display: flex; flex-direction: column; gap: 6px; min-width: 175px; padding-left: 20px; border-left: 1px solid var(--border);">
          <div style="display: flex; justify-content: space-between; font-size: 11px;">
            <span style="color: var(--text-dim);">Glove Link:</span>
            <span id="glove-sync-pill" style="color: var(--accent-cyan); font-weight: 600;">Standby</span>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 11px;">
            <span style="color: var(--text-dim);">Belt Link:</span>
            <span id="belt-sync-pill" style="color: var(--accent-cyan); font-weight: 600;">Waiting</span>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 11px;">
            <span style="color: var(--text-dim);">Fusion State:</span>
            <span id="fusion-sync-pill" style="color: var(--text-dim); font-weight: 500;">Standalone</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 2. Dual-Node Sensor Data Grid (Glove on Left, Belt on Right) -->
    <div class="dual-node-grid">
      
      <!-- CARD 1: Smart Glove Sensor Data -->
      <div class="card" style="border: 1px solid rgba(6, 182, 212, 0.35); box-shadow: 0 4px 20px rgba(6, 182, 212, 0.06); display: flex; flex-direction: column;">
        <div class="card-header" style="margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
          <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 32px; height: 32px; border-radius: 8px; background: rgba(6, 182, 212, 0.2); display: flex; align-items: center; justify-content: center; font-size: 16px;">ðŸ§¤</div>
            <div>
              <span class="card-title" style="color: var(--accent-cyan); font-size: 14px; font-weight: 700;">Smart Glove Sensor Data</span>
              <div style="font-size: 11px; color: var(--text-dim);">Wrist Node â€¢ HEALTH_GLOVE_01</div>
            </div>
          </div>
          <div style="display: flex; align-items: center; gap: 6px;">
            <span class="badge-pill cyan" id="glove-pkt-badge">0 packets</span>
            <span class="badge-pill" id="glove-finger-badge" style="background: rgba(255,255,255,0.08); color: var(--text-dim);">No Finger</span>
          </div>
        </div>

        <!-- Glove Primary Vitals -->
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 14px;">
          <!-- Heart Rate -->
          <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid var(--border); border-radius: 12px; padding: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <span style="font-size: 11px; color: var(--text-dim); text-transform: uppercase; font-weight: 600;">Heart Rate</span>
              <span id="hr-src" style="font-size: 10px; color: var(--accent-cyan);">MAX30100</span>
            </div>
            <div style="display: flex; align-items: baseline;">
              <span class="vital-value" id="val-hr" style="font-size: 26px;">--</span>
              <span class="vital-unit" style="font-size: 12px;">BPM</span>
            </div>
            <div id="hr-sub" style="font-size: 10px; color: var(--text-dim); margin-top: 4px;">Pulse standby</div>
          </div>

          <!-- SpO2 -->
          <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid var(--border); border-radius: 12px; padding: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <span style="font-size: 11px; color: var(--text-dim); text-transform: uppercase; font-weight: 600;">Blood Oxygen</span>
              <span id="spo2-src" style="font-size: 10px; color: var(--accent-cyan);">MAX30100</span>
            </div>
            <div style="display: flex; align-items: baseline;">
              <span class="vital-value" id="val-spo2" style="font-size: 26px;">--</span>
              <span class="vital-unit" style="font-size: 12px;">%</span>
            </div>
            <div id="spo2-sub" style="font-size: 10px; color: var(--text-dim); margin-top: 4px;">SpOâ‚‚ saturation</div>
          </div>

          <!-- Temperature -->
          <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid var(--border); border-radius: 12px; padding: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <span style="font-size: 11px; color: var(--text-dim); text-transform: uppercase; font-weight: 600;">Body Temp</span>
              <span id="temp-src" style="font-size: 10px; color: var(--accent-cyan);">MLX90614</span>
            </div>
            <div style="display: flex; align-items: baseline;">
              <span class="vital-value" id="val-temp" style="font-size: 26px;">--</span>
              <span class="vital-unit" style="font-size: 12px;">Â°C</span>
            </div>
            <div id="temp-sub" style="font-size: 10px; color: var(--text-dim); margin-top: 4px;">Skin infrared</div>
          </div>
        </div>

        <!-- Glove Secondary Sensors Grid -->
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; flex: 1;">
          <!-- GSR & Stress -->
          <div class="sub-sensor-tile">
            <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; margin-bottom: 4px;">GSR Conductance (Stress)</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: #a7f3d0;">
              <span id="glove-gsr-val">--</span> <span style="font-size: 10px; color: var(--text-dim); font-weight: 400;">ADC</span>
              <span style="margin: 0 4px; color: rgba(255,255,255,0.2);">|</span>
              <span id="glove-gsr-volt">--</span> <span style="font-size: 10px; color: var(--text-dim); font-weight: 400;">V</span>
            </div>
          </div>

          <!-- AD8232 ECG -->
          <div class="sub-sensor-tile">
            <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; margin-bottom: 4px;">AD8232 ECG Stream</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: #38bdf8;">
              <span id="glove-ecg-rate">250 Hz</span> <span style="font-size: 10px; color: var(--text-dim); font-weight: 400;">(<span id="glove-ecg-count">0</span> samples)</span>
            </div>
          </div>

          <!-- Hand Tremor IMU -->
          <div class="sub-sensor-tile">
            <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center;">
              <span>Hand IMU (50 Hz Parkinson's)</span>
              <span id="glove-imu-status" style="color: var(--accent-green); font-size: 10px; text-transform: none; font-weight: 500;">Steady</span>
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #cbd5e1;">
              Accel: <span id="glove-ax">0.00</span>, <span id="glove-ay">0.00</span>, <span id="glove-az">9.81</span> m/sÂ²
            </div>
          </div>

          <!-- GPS Location -->
          <div class="sub-sensor-tile">
            <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; margin-bottom: 4px;">NEO-M8N GPS Fix</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #cbd5e1;">
              <span id="glove-gps-status" style="color: var(--accent-amber);">Searching</span> | <span id="glove-gps-sats">0</span> Sats
            </div>
          </div>
        </div>
      </div>

      <!-- CARD 2: Belt Sensor Data -->
      <div class="card" style="border: 1px solid rgba(139, 92, 246, 0.35); box-shadow: 0 4px 20px rgba(139, 92, 246, 0.06); display: flex; flex-direction: column;">
        <div class="card-header" style="margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
          <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 32px; height: 32px; border-radius: 8px; background: rgba(139, 92, 246, 0.2); display: flex; align-items: center; justify-content: center; font-size: 16px;">ðŸ¥‹</div>
            <div>
              <span class="card-title" style="color: #c084fc; font-size: 14px; font-weight: 700;">Smart Belt Sensor Data</span>
              <div style="font-size: 11px; color: var(--text-dim);">Waist / Spine Node â€¢ WAIST_NODE</div>
            </div>
          </div>
          <div style="display: flex; align-items: center; gap: 6px;">
            <span class="badge-pill purple" id="belt-pkt-badge">0 packets</span>
            <span class="badge-pill green" id="belt-rate-badge">2 Hz Active</span>
          </div>
        </div>

        <!-- Belt Primary Metrics -->
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 14px;">
          <!-- Posture -->
          <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid var(--border); border-radius: 12px; padding: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <span style="font-size: 11px; color: var(--text-dim); text-transform: uppercase; font-weight: 600;">Posture</span>
              <span id="belt-posture-icon" style="font-size: 14px;">ðŸ§</span>
            </div>
            <div style="display: flex; align-items: baseline;">
              <span class="vital-value" id="belt-val-posture" style="font-size: 20px; color: #c084fc;">STANDING</span>
            </div>
            <div id="belt-posture-sub" style="font-size: 10px; color: var(--text-dim); margin-top: 4px;">Torso orientation</div>
          </div>

          <!-- Fall Status -->
          <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid var(--border); border-radius: 12px; padding: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <span style="font-size: 11px; color: var(--text-dim); text-transform: uppercase; font-weight: 600;">Fall Status</span>
              <span id="belt-fall-icon" style="font-size: 14px;">ðŸ›¡ï¸</span>
            </div>
            <div style="display: flex; align-items: baseline;">
              <span class="vital-value" id="belt-val-fall" style="font-size: 20px; color: var(--accent-green);">SAFE</span>
            </div>
            <div id="belt-fall-sub" style="font-size: 10px; color: var(--text-dim); margin-top: 4px;">Impact & latch monitor</div>
          </div>

          <!-- Acceleration Magnitude -->
          <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid var(--border); border-radius: 12px; padding: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
              <span style="font-size: 11px; color: var(--text-dim); text-transform: uppercase; font-weight: 600;">Magnitude</span>
              <span style="font-size: 10px; color: #c084fc;">MPU6050</span>
            </div>
            <div style="display: flex; align-items: baseline;">
              <span class="vital-value" id="belt-val-mag" style="font-size: 20px; color: #f9fafb;">9.81</span>
              <span class="vital-unit" style="font-size: 11px;">m/sÂ²</span>
            </div>
            <div id="belt-mag-sub" style="font-size: 10px; color: var(--text-dim); margin-top: 4px;">Live torso motion</div>
          </div>
        </div>

        <!-- Belt Secondary Sensors Grid -->
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; flex: 1;">
          <!-- Tilt Angle -->
          <div class="sub-sensor-tile">
            <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; margin-bottom: 4px;">Torso Tilt Angle</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: #c084fc;">
              <span id="belt-val-tilt">0.0</span>Â° <span style="font-size: 10px; color: var(--text-dim); font-weight: 400;">(<span id="belt-tilt-desc">Upright</span>)</span>
            </div>
          </div>

          <!-- Activity Variance -->
          <div class="sub-sensor-tile">
            <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; margin-bottom: 4px;">Activity Variance</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: #38bdf8;">
              <span id="belt-val-act">0.00</span> <span style="font-size: 10px; color: var(--text-dim); font-weight: 400;">(<span id="belt-act-desc">Resting</span>)</span>
            </div>
          </div>

          <!-- 3-Axis Acceleration -->
          <div class="sub-sensor-tile">
            <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; margin-bottom: 4px;">Torso Accel (m/sÂ²)</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #cbd5e1;">
              X:<span id="belt-ax">0.00</span> Y:<span id="belt-ay">0.00</span> Z:<span id="belt-az">9.81</span>
            </div>
          </div>

          <!-- 3-Axis Gyroscope -->
          <div class="sub-sensor-tile">
            <div style="font-size: 10px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; margin-bottom: 4px;">Torso Gyro (Â°/s)</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #cbd5e1;">
              X:<span id="belt-gx">0.0</span> Y:<span id="belt-gy">0.0</span> Z:<span id="belt-gz">0.0</span>
            </div>
          </div>
        </div>
      </div>

    </div>

    <!-- Glove Raw Hardware Telemetry Section (Above AI Screening & Engines) -->
    <div class="card" style="margin-bottom: 24px; border: 1px solid rgba(6, 182, 212, 0.35); box-shadow: 0 4px 20px rgba(6, 182, 212, 0.05);">
      <div class="card-header" style="flex-wrap: wrap; gap: 12px;">
        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
          <span class="card-title" style="color: var(--accent-cyan); font-size: 14px;">Glove Raw Hardware Telemetry</span>
          <span class="badge-pill green" id="raw-node-badge">HEALTH_GLOVE_01</span>
          <span class="badge-pill cyan" id="raw-pkt-badge">0 packets</span>
          <span style="font-size: 12px; color: var(--text-dim); font-family: 'JetBrains Mono';" id="raw-time-badge">Waiting for hardware...</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
          <button class="btn-action" id="btn-pause-raw" onclick="togglePauseRaw()">â¸ Freeze Feed</button>
          <button class="btn-action" onclick="copyRawJson()">ðŸ“‹ Copy JSON</button>
          <button class="btn-action" id="btn-toggle-compact" onclick="toggleCompactJson()">ðŸ—œ Compact Arrays</button>
        </div>
      </div>

      <!-- Tab Navigation (ECG Oscilloscope First) -->
      <div class="tabs-nav">
        <button class="tab-btn active" id="tab-btn-ecg" onclick="switchRawTab('ecg')">ðŸ“ˆ ECG Oscilloscope (250Hz)</button>
        <button class="tab-btn" id="tab-btn-sensors" onclick="switchRawTab('sensors')">ðŸ“Š Sensors Breakdown</button>
        <button class="tab-btn" id="tab-btn-json" onclick="switchRawTab('json')">ðŸ“„ Raw Ingested JSON</button>
      </div>

      <!-- Tab Content: ECG Oscilloscope (Active First Tab) -->
      <div id="tab-content-ecg" style="display: block;">
        <div style="background: #040810; border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 12px; padding: 12px; position: relative;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 12px; font-family: 'JetBrains Mono'; color: #38bdf8;">RAW AD8232 ECG BUFFER (250 SAMPLES / PACKET @ 250 Hz)</span>
            <span style="font-size: 11px; color: var(--text-dim); font-family: 'JetBrains Mono';" id="ecg-canvas-stats">Waiting for samples...</span>
          </div>
          <canvas id="rawEcgCanvas" width="900" height="150" style="width: 100%; height: 150px; display: block; border-radius: 8px;"></canvas>
        </div>
      </div>

      <!-- Tab Content: Sensors Breakdown -->
      <div id="tab-content-sensors" style="display: none;">
        <div class="raw-sensors-grid">
          <!-- MAX30100 -->
          <div class="sensor-tile">
            <div class="sensor-tile-title">
              <span>MAX30100 Pulse Oximeter</span>
              <span class="badge-pill cyan">I2C 0x57</span>
            </div>
            <div class="sensor-tile-val">
              <span id="raw-hr">--</span> <span style="font-size:12px; color:var(--text-dim);">BPM</span>
              <span style="margin: 0 8px; color:rgba(255,255,255,0.2);">|</span>
              <span id="raw-spo2">--</span> <span style="font-size:12px; color:var(--text-dim);">% SpOâ‚‚</span>
            </div>
            <div class="sensor-tile-sub" id="raw-ppg-status">Status: Standby</div>
          </div>

          <!-- MLX90614 -->
          <div class="sensor-tile">
            <div class="sensor-tile-title">
              <span>MLX90614 Contactless IR Temp</span>
              <span class="badge-pill cyan">I2C 0x5A</span>
            </div>
            <div class="sensor-tile-val">
              <span id="raw-temp">--</span> <span style="font-size:13px; color:var(--text-dim);">Â°C</span>
            </div>
            <div class="sensor-tile-sub" id="raw-temp-status">Target: Skin surface</div>
          </div>

          <!-- GSR -->
          <div class="sensor-tile">
            <div class="sensor-tile-title">
              <span>GSR (Skin Conductance)</span>
              <span class="badge-pill cyan">GPIO 34 ADC1</span>
            </div>
            <div class="sensor-tile-val">
              <span id="raw-gsr">--</span> <span style="font-size:12px; color:var(--text-dim);">/ 4095</span>
            </div>
            <div class="sensor-tile-sub">
              Est. Voltage: <span id="raw-gsr-volt">--</span> V
            </div>
          </div>

          <!-- NEO-M8N GPS -->
          <div class="sensor-tile">
            <div class="sensor-tile-title">
              <span>NEO-M8N GPS Module</span>
              <span class="badge-pill amber" id="raw-gps-badge">NO FIX</span>
            </div>
            <div class="sensor-tile-val" style="font-size: 13px; font-weight: 500;">
              Lat: <span id="raw-gps-lat">--</span><br>Lng: <span id="raw-gps-lng">--</span>
            </div>
            <div class="sensor-tile-sub">
              Sats: <span id="raw-gps-sats">0</span> | Speed: <span id="raw-gps-speed">0.0</span> km/h
            </div>
          </div>

          <!-- MPU6050 Motion -->
          <div class="sensor-tile">
            <div class="sensor-tile-title">
              <span>MPU6050 6-Axis Motion IMU</span>
              <span class="badge-pill cyan" id="raw-imu-badge">50 Hz</span>
            </div>
            <div class="sensor-tile-sub" style="font-size: 11px; margin-bottom: 2px;">
              Accel (m/sÂ²): X:<span id="raw-ax">--</span> Y:<span id="raw-ay">--</span> Z:<span id="raw-az">--</span>
            </div>
            <div class="sensor-tile-sub" style="font-size: 11px; margin-bottom: 2px;">
              Gyro (Â°/s): X:<span id="raw-gx">--</span> Y:<span id="raw-gy">--</span> Z:<span id="raw-gz">--</span>
            </div>
            <div class="sensor-tile-sub">
              Buffer: <span id="raw-imu-count">0</span> samples | SOS: <span id="raw-sos" style="font-weight:600; color:var(--text-dim);">false</span>
            </div>
          </div>

          <!-- AD8232 ECG -->
          <div class="sensor-tile">
            <div class="sensor-tile-title">
              <span>AD8232 ECG Heart Monitor</span>
              <span class="badge-pill cyan" id="raw-ecg-rate-badge">250 Hz</span>
            </div>
            <div class="sensor-tile-val">
              <span id="raw-ecg-last">--</span> <span style="font-size:12px; color:var(--text-dim);">ADC</span>
            </div>
            <div class="sensor-tile-sub">
              Buffer: <span id="raw-ecg-samples-count">0</span> samples | Range: [<span id="raw-ecg-min">--</span>, <span id="raw-ecg-max">--</span>]
            </div>
          </div>
        </div>
      </div>

      <!-- Tab Content: Raw JSON Viewer -->
      <div id="tab-content-json" style="display: none;">
        <div class="raw-json-viewer" id="raw-json-display">// Waiting for raw telemetry packet from glove...</div>
      </div>
    </div>

    <!-- AI Screenings and Advanced Engines -->
    <div class="grid" style="grid-template-columns: 1fr 1fr;">
      <div class="card">
        <div class="card-header">
          <span class="card-title">AI Screening Pipelines (0-100%)</span>
        </div>
        <div class="screening-row">
          <span>Arrhythmia Likelihood</span>
          <div style="display:flex; align-items:center; gap:10px;">
            <div class="prob-bar-container"><div id="bar-arrhythmia" class="prob-bar"></div></div>
            <span id="txt-arrhythmia" style="font-family: 'JetBrains Mono'; font-size:13px;">--%</span>
          </div>
        </div>
        <div class="screening-row">
          <span>Parkinson's Likelihood</span>
          <div style="display:flex; align-items:center; gap:10px;">
            <div class="prob-bar-container"><div id="bar-parkinsons" class="prob-bar"></div></div>
            <span id="txt-parkinsons" style="font-family: 'JetBrains Mono'; font-size:13px;">--%</span>
          </div>
        </div>
        <div class="screening-row">
          <span>Hypoxemia Likelihood</span>
          <div style="display:flex; align-items:center; gap:10px;">
            <div class="prob-bar-container"><div id="bar-hypoxemia" class="prob-bar"></div></div>
            <span id="txt-hypoxemia" style="font-family: 'JetBrains Mono'; font-size:13px;">--%</span>
          </div>
        </div>
        <div class="screening-row">
          <span>Sepsis Likelihood</span>
          <div style="display:flex; align-items:center; gap:10px;">
            <div class="prob-bar-container"><div id="bar-sepsis" class="prob-bar"></div></div>
            <span id="txt-sepsis" style="font-family: 'JetBrains Mono'; font-size:13px;">--%</span>
          </div>
        </div>
        <div class="screening-row">
          <span>Stress Likelihood</span>
          <div style="display:flex; align-items:center; gap:10px;">
            <div class="prob-bar-container"><div id="bar-stress" class="prob-bar"></div></div>
            <span id="txt-stress" style="font-family: 'JetBrains Mono'; font-size:13px;">--%</span>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">Advanced Detection Engines</span>
        </div>
        <div class="screening-row">
          <span>Dual-Node Sensor Fusion</span>
          <span id="txt-fusion" style="color: var(--accent-cyan); font-weight:600; font-size:13px;">No Waist Data</span>
        </div>
        <div class="screening-row">
          <span>Multi-Sensor Correlation</span>
          <span id="txt-correlation" style="color: var(--accent-green); font-weight:600; font-size:13px;">Score: 0</span>
        </div>
        <div class="screening-row">
          <span>Early Sepsis Warning</span>
          <span id="txt-early-sepsis" style="font-size:13px; color: var(--text-dim);">Normal</span>
        </div>
        <div class="screening-row">
          <span>Silent Hypoxia ETA</span>
          <span id="txt-silent-hypoxia" style="font-size:13px; color: var(--text-dim);">Stable</span>
        </div>
        <div class="screening-row">
          <span>Silent SOS Gesture</span>
          <span id="txt-sos" style="font-size:13px; color: var(--text-dim);">Ready</span>
        </div>
      </div>
    </div>

    <!-- Disease & ML Simulation Control Panel -->
    <div class="card sim-card" id="simulation-panel">
      <div class="sim-header">
        <div class="sim-title-group">
          <div class="sim-icon">âš¡</div>
          <div>
            <div class="sim-heading">Disease & ML Pipeline Simulator</div>
            <div class="sim-subheading">Inject clinical disease scenarios â€¢ Physical glove data ignored while simulating</div>
          </div>
        </div>
        <div class="sim-status-wrapper">
          <div id="sim-status-badge" class="sim-badge-idle">
            <span class="sim-dot-idle" id="sim-status-dot"></span>
            <span id="sim-status-label">Live Glove Mode</span>
          </div>
          <button id="sim-stop-btn" class="sim-btn-stop" onclick="stopSimulation()" style="display: none;">
            â¹ Stop Simulation
          </button>
        </div>
      </div>

      <!-- Progress bar shown when simulating -->
      <div id="sim-progress-container" class="sim-progress-container" style="display: none;">
        <div class="sim-progress-info">
          <span id="sim-progress-text">Simulating...</span>
          <span id="sim-progress-pct" style="font-family: 'JetBrains Mono', monospace;">0%</span>
        </div>
        <div class="sim-progress-track">
          <div id="sim-progress-bar" class="sim-progress-fill" style="width: 0%;"></div>
        </div>
      </div>

      <!-- Disease Selection Buttons -->
      <div class="sim-grid">
        <button class="sim-btn" id="btn-sim-arrhythmia" onclick="startSimulation('arrhythmia')">
          <div class="sim-btn-top">
            <span class="sim-btn-icon">ðŸ©º</span>
            <span class="sim-btn-tag" style="color: #fca5a5; background: rgba(239,68,68,0.15);">Severity 2</span>
          </div>
          <div class="sim-btn-name">Arrhythmia (VEB / PVC)</div>
          <div class="sim-btn-desc">Ventricular Ectopic Beat (Severity 2 Alert)</div>
        </button>

        <button class="sim-btn" id="btn-sim-parkinsons" onclick="startSimulation('parkinsons')">
          <div class="sim-btn-top">
            <span class="sim-btn-icon">ðŸ§ </span>
            <span class="sim-btn-tag">3-Axis Accel</span>
          </div>
          <div class="sim-btn-name">Parkinson's</div>
          <div class="sim-btn-desc">4.5â€“5.0 Hz Resting Tremor</div>
        </button>

        <button class="sim-btn" id="btn-sim-hypoxemia" onclick="startSimulation('hypoxemia')">
          <div class="sim-btn-top">
            <span class="sim-btn-icon">ðŸ«</span>
            <span class="sim-btn-tag">100-Avg SpOâ‚‚</span>
          </div>
          <div class="sim-btn-name">Hypoxemia</div>
          <div class="sim-btn-desc">Severe Desaturation (~87% SpOâ‚‚)</div>
        </button>

        <button class="sim-btn" id="btn-sim-sepsis" onclick="startSimulation('sepsis')">
          <div class="sim-btn-top">
            <span class="sim-btn-icon">ðŸŒ¡ï¸</span>
            <span class="sim-btn-tag">Multi-Vital SIRS</span>
          </div>
          <div class="sim-btn-name">Sepsis</div>
          <div class="sim-btn-desc">38.8Â°C Fever + Tachycardia + Low Oâ‚‚</div>
        </button>

        <button class="sim-btn" id="btn-sim-stress" onclick="startSimulation('stress')">
          <div class="sim-btn-top">
            <span class="sim-btn-icon">âš¡</span>
            <span class="sim-btn-tag">GSR + Vitals</span>
          </div>
          <div class="sim-btn-name">Acute Stress</div>
          <div class="sim-btn-desc">High Skin Conductance & HR</div>
        </button>

        <button class="sim-btn" id="btn-sim-silent_hypoxia" onclick="startSimulation('silent_hypoxia')">
          <div class="sim-btn-top">
            <span class="sim-btn-icon">ðŸ“‰</span>
            <span class="sim-btn-tag">Theilâ€“Sen Trend</span>
          </div>
          <div class="sim-btn-name">Silent Hypoxia</div>
          <div class="sim-btn-desc">Rapid SpOâ‚‚ Velocity Decline</div>
        </button>
      </div>
    </div>

    <!-- Live Ingestion Log -->
    <div class="card" style="margin-bottom: 24px;">
      <div class="card-header">
        <span class="card-title">Live Ingestion Event Log</span>
      </div>
      <div class="raw-log" id="event-log">
        [System] Waiting for ESP32 telemetry packets...
      </div>
    </div>

    <!-- Network Configuration Card (Bottom) -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Local LAN Endpoints for ESP32 Firmware</span>
      </div>
      <p style="font-size: 14px; color: var(--text-dim); margin-bottom: 8px;">
        Use one of the following IP URLs in your ESP32 Arduino sketches (<code>glove/glove.ino</code> / <code>waist_fall_node.ino</code>):
      </p>
      <div class="endpoints-box" id="ip-endpoints">
        Loading local IP addresses...
      </div>
    </div>
  </div>

  <script>
    // Glove Raw Inspector State
    let latestRawPacket = null;
    let isRawPaused = false;
    let compactArrays = true;
    let rawPacketCount = 0;
    let lastRawPacketTime = null;
    let activeRawTab = 'ecg';

    function switchRawTab(tab) {
      activeRawTab = tab;
      document.getElementById('tab-btn-sensors').classList.toggle('active', tab === 'sensors');
      document.getElementById('tab-btn-json').classList.toggle('active', tab === 'json');
      document.getElementById('tab-btn-ecg').classList.toggle('active', tab === 'ecg');

      document.getElementById('tab-content-sensors').style.display = (tab === 'sensors') ? 'block' : 'none';
      document.getElementById('tab-content-json').style.display = (tab === 'json') ? 'block' : 'none';
      document.getElementById('tab-content-ecg').style.display = (tab === 'ecg') ? 'block' : 'none';

      if (tab === 'ecg' && latestRawPacket && latestRawPacket.ecg_samples) {
        drawEcgWaveform(latestRawPacket.ecg_samples);
      }
    }

    function togglePauseRaw() {
      isRawPaused = !isRawPaused;
      const btn = document.getElementById('btn-pause-raw');
      if (isRawPaused) {
        btn.textContent = 'â–¶ Resume Feed';
        btn.classList.add('paused');
      } else {
        btn.textContent = 'â¸ Freeze Feed';
        btn.classList.remove('paused');
        if (latestRawPacket) updateRawInspector(latestRawPacket, true);
      }
    }

    function toggleCompactJson() {
      compactArrays = !compactArrays;
      document.getElementById('btn-toggle-compact').textContent = compactArrays ? 'ðŸ—œ Compact Arrays' : 'ðŸ“œ Full Arrays';
      if (latestRawPacket) renderRawJson(latestRawPacket);
    }

    function copyRawJson() {
      if (!latestRawPacket) return;
      navigator.clipboard.writeText(JSON.stringify(latestRawPacket, null, 2)).then(() => {
        const btn = event.currentTarget || event.target;
        const oldText = btn.textContent;
        btn.textContent = 'âœ“ Copied!';
        setTimeout(() => { btn.textContent = oldText; }, 2000);
      });
    }

    function renderRawJson(raw) {
      const viewer = document.getElementById('raw-json-display');
      if (!raw) {
        viewer.textContent = '// No packet received yet';
        return;
      }
      if (compactArrays) {
        const clone = {};
        for (const [k, v] of Object.entries(raw)) {
          if (Array.isArray(v) && v.length > 10) {
            clone[k] = `[Array of ${v.length} items: ${v.slice(0, 5).join(', ')} ... +${v.length - 5} more]`;
          } else {
            clone[k] = v;
          }
        }
        viewer.textContent = JSON.stringify(clone, null, 2);
      } else {
        viewer.textContent = JSON.stringify(raw, null, 2);
      }
    }

    function drawEcgWaveform(samples) {
      const canvas = document.getElementById('rawEcgCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);

      // Grid background
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.08)';
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 30) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += 25) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Center baseline
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.2)';
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();

      if (!samples || !samples.length) {
        ctx.fillStyle = 'rgba(156, 163, 175, 0.6)';
        ctx.font = "13px 'JetBrains Mono', monospace";
        ctx.textAlign = 'center';
        ctx.fillText('Waiting for AD8232 ECG streaming buffer from glove...', w / 2, h / 2 + 5);
        return;
      }

      let min = samples[0];
      let max = samples[0];
      for (let i = 1; i < samples.length; i++) {
        if (samples[i] < min) min = samples[i];
        if (samples[i] > max) max = samples[i];
      }
      const range = (max - min) || 1;
      const padding = 15;
      const usableH = h - (padding * 2);

      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 1.8;
      ctx.shadowColor = '#10b981';
      ctx.shadowBlur = 6;
      ctx.beginPath();

      const step = w / (samples.length - 1 || 1);
      for (let i = 0; i < samples.length; i++) {
        const x = i * step;
        const norm = (samples[i] - min) / range;
        const y = (h - padding) - (norm * usableH);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      const statsEl = document.getElementById('ecg-canvas-stats');
      if (statsEl) {
        statsEl.textContent = `${samples.length} samples | Min: ${min} | Max: ${max} | Peak-to-Peak: ${max - min}`;
      }
    }

    function updateRawInspector(raw, force) {
      if (!raw) return;
      if (isRawPaused && !force) return;

      latestRawPacket = raw;
      rawPacketCount++;
      lastRawPacketTime = Date.now();

      if (raw.device_id) document.getElementById('raw-node-badge').textContent = raw.device_id;
      document.getElementById('raw-pkt-badge').textContent = `${rawPacketCount} packets`;
      document.getElementById('raw-time-badge').textContent = 'Just received';

      // 1. MAX30100
      if (raw.heart_rate !== undefined) {
        document.getElementById('raw-hr').textContent = Number(raw.heart_rate).toFixed(1);
      }
      if (raw.spo2 !== undefined) {
        document.getElementById('raw-spo2').textContent = raw.spo2;
      }
      const ppgStatus = document.getElementById('raw-ppg-status');
      if (raw.heart_rate > 0) {
        ppgStatus.textContent = 'Status: Pulse active (finger detected)';
        ppgStatus.style.color = '#34d399';
      } else {
        ppgStatus.textContent = 'Status: Standby / Finger off';
        ppgStatus.style.color = 'var(--text-dim)';
      }

      // 2. MLX90614
      if (raw.temperature !== undefined) {
        document.getElementById('raw-temp').textContent = Number(raw.temperature).toFixed(2);
      }
      const tempStatus = document.getElementById('raw-temp-status');
      if (raw.temperature > 0) {
        tempStatus.textContent = `Target: ${Number(raw.temperature).toFixed(1)}Â°C (Online)`;
        tempStatus.style.color = '#38bdf8';
      }

      // 3. GSR
      if (raw.gsr_raw !== undefined) {
        document.getElementById('raw-gsr').textContent = raw.gsr_raw;
        document.getElementById('raw-gsr-volt').textContent = (raw.gsr_raw / 4095.0 * 3.3).toFixed(2);
      }

      // 4. GPS
      const gpsBadge = document.getElementById('raw-gps-badge');
      if (raw.gps_valid) {
        gpsBadge.textContent = 'FIX VALID';
        gpsBadge.className = 'badge-pill green';
      } else {
        gpsBadge.textContent = 'SEARCHING';
        gpsBadge.className = 'badge-pill amber';
      }
      if (raw.latitude !== undefined && raw.latitude !== 0) {
        document.getElementById('raw-gps-lat').textContent = Number(raw.latitude).toFixed(5);
      }
      if (raw.longitude !== undefined && raw.longitude !== 0) {
        document.getElementById('raw-gps-lng').textContent = Number(raw.longitude).toFixed(5);
      }
      document.getElementById('raw-gps-sats').textContent = raw.satellites || 0;
      document.getElementById('raw-gps-speed').textContent = raw.speed_kmph !== undefined ? Number(raw.speed_kmph).toFixed(1) : '0.0';

      // 5. IMU
      const axArr = raw.ax_samples || (raw.ax !== undefined ? [raw.ax] : []);
      const ayArr = raw.ay_samples || (raw.ay !== undefined ? [raw.ay] : []);
      const azArr = raw.az_samples || (raw.az !== undefined ? [raw.az] : []);
      const gxArr = raw.gx_samples || (raw.gx !== undefined ? [raw.gx] : []);
      const gyArr = raw.gy_samples || (raw.gy !== undefined ? [raw.gy] : []);
      const gzArr = raw.gz_samples || (raw.gz !== undefined ? [raw.gz] : []);

      if (axArr.length) document.getElementById('raw-ax').textContent = Number(axArr[axArr.length - 1]).toFixed(2);
      if (ayArr.length) document.getElementById('raw-ay').textContent = Number(ayArr[ayArr.length - 1]).toFixed(2);
      if (azArr.length) document.getElementById('raw-az').textContent = Number(azArr[azArr.length - 1]).toFixed(2);
      if (gxArr.length) document.getElementById('raw-gx').textContent = Number(gxArr[gxArr.length - 1]).toFixed(2);
      if (gyArr.length) document.getElementById('raw-gy').textContent = Number(gyArr[gyArr.length - 1]).toFixed(2);
      if (gzArr.length) document.getElementById('raw-gz').textContent = Number(gzArr[gzArr.length - 1]).toFixed(2);

      document.getElementById('raw-imu-count').textContent = axArr.length || 0;
      const sosEl = document.getElementById('raw-sos');
      if (raw.sos_gesture) {
        sosEl.textContent = 'TRIGGERED (SHAKE)';
        sosEl.style.color = 'var(--accent-red)';
      } else {
        sosEl.textContent = 'false (Resting)';
        sosEl.style.color = 'var(--text-dim)';
      }

      // 6. ECG
      const ecgArr = raw.ecg_samples || [];
      document.getElementById('raw-ecg-samples-count').textContent = ecgArr.length;
      if (ecgArr.length) {
        document.getElementById('raw-ecg-last').textContent = ecgArr[ecgArr.length - 1];
        let minE = ecgArr[0], maxE = ecgArr[0];
        for (let i = 1; i < ecgArr.length; i++) {
          if (ecgArr[i] < minE) minE = ecgArr[i];
          if (ecgArr[i] > maxE) maxE = ecgArr[i];
        }
        document.getElementById('raw-ecg-min').textContent = minE;
        document.getElementById('raw-ecg-max').textContent = maxE;
        if (activeRawTab === 'ecg') {
          drawEcgWaveform(ecgArr);
        }
      }

      renderRawJson(raw);
    }

    // Packet time elapsed ticker
    setInterval(() => {
      if (!lastRawPacketTime) return;
      const sec = Math.floor((Date.now() - lastRawPacketTime) / 1000);
      const badge = document.getElementById('raw-time-badge');
      if (sec < 2) badge.textContent = 'Just received';
      else if (sec < 60) badge.textContent = `${sec}s ago`;
      else badge.textContent = `${Math.floor(sec / 60)}m ${sec % 60}s ago`;
    }, 1000);

    // Initialize ECG oscilloscope grid
    drawEcgWaveform([]);

    // Fetch server status and IP addresses on load
    fetch('/api/status')
      .then(r => r.json())
      .then(data => {
        const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
        const list = data.local_ip_addresses.map(ip => `POST http://${ip}:${port}/api/telemetry`).join('<br>');
        document.getElementById('ip-endpoints').innerHTML = list;
      });

    let beltPktCount = 0;
    let glovePktCount = 0;

    function updateVitalsCards(pkt) {
      if (!pkt) return;
      glovePktCount++;
      const gBadge = document.getElementById('glove-pkt-badge');
      if (gBadge) gBadge.textContent = `${glovePktCount} packets`;

      const gSync = document.getElementById('glove-sync-pill');
      if (gSync) {
        gSync.textContent = 'Online';
        gSync.style.color = 'var(--accent-green)';
      }

      const fingerOnNow = pkt.finger_detected === false ? false :
                          ((pkt.spo2 > 0) || (pkt.heart_rate > 0));

      const fingerBadge = document.getElementById('glove-finger-badge');
      if (fingerBadge) {
        if (fingerOnNow) {
          fingerBadge.textContent = 'Finger Active';
          fingerBadge.style.background = 'rgba(16, 185, 129, 0.15)';
          fingerBadge.style.color = 'var(--accent-green)';
        } else {
          fingerBadge.textContent = 'No Finger';
          fingerBadge.style.background = 'rgba(255, 255, 255, 0.08)';
          fingerBadge.style.color = 'var(--text-dim)';
        }
      }

      // Heart Rate
      if (pkt.heart_rate !== undefined && pkt.heart_rate !== null) {
        document.getElementById('val-hr').textContent = fingerOnNow ? Number(pkt.heart_rate).toFixed(1) : '--';
        const hrSrc = document.getElementById('hr-src');
        const hrSub = document.getElementById('hr-sub');
        if (hrSrc) {
          hrSrc.textContent = pkt.heart_rate > 0 ? 'Pulse Active' : 'Standby / Finger Off';
          hrSrc.style.color = pkt.heart_rate > 0 ? 'var(--accent-green)' : 'var(--text-dim)';
        }
        if (hrSub) {
          hrSub.textContent = pkt.heart_rate > 0 ? `${Number(pkt.heart_rate).toFixed(0)} BPM detected` : 'Waiting for finger...';
        }
      }

      // SpO2
      if (pkt.spo2 !== undefined && pkt.spo2 !== null) {
        document.getElementById('val-spo2').textContent = fingerOnNow ? pkt.spo2 : '--';
        const spo2Src = document.getElementById('spo2-src');
        const spo2Sub = document.getElementById('spo2-sub');
        if (spo2Src) {
          spo2Src.textContent = pkt.spo2 > 0 ? 'Pulse Active' : 'Standby / Finger Off';
          spo2Src.style.color = pkt.spo2 > 0 ? 'var(--accent-green)' : 'var(--text-dim)';
        }
        if (spo2Sub) {
          spo2Sub.textContent = pkt.spo2 > 0 ? `${pkt.spo2}% SpOâ‚‚ reading` : 'Waiting for finger...';
        }
      }

      // Body Temperature
      if (pkt.temperature !== undefined && pkt.temperature !== null) {
        document.getElementById('val-temp').textContent = fingerOnNow ? Number(pkt.temperature).toFixed(1) : '--';
        const tempSrc = document.getElementById('temp-src');
        const tempSub = document.getElementById('temp-sub');
        if (tempSrc) {
          tempSrc.textContent = fingerOnNow ? 'MLX90614 Active' : 'Standby / Finger Off';
          tempSrc.style.color = fingerOnNow ? 'var(--accent-green)' : 'var(--text-dim)';
        }
        if (tempSub) {
          tempSub.textContent = fingerOnNow ? `${Number(pkt.temperature).toFixed(1)}Â°C IR skin surface` : 'Infrared target standby';
        }
      }

      // Glove Secondary Sensors
      if (pkt.gsr_raw !== undefined) {
        const gsrEl = document.getElementById('glove-gsr-val');
        const voltEl = document.getElementById('glove-gsr-volt');
        if (gsrEl) gsrEl.textContent = pkt.gsr_raw;
        if (voltEl) voltEl.textContent = (pkt.gsr_raw / 4095.0 * 3.3).toFixed(2);
      }

      if (pkt.ecg_samples && Array.isArray(pkt.ecg_samples)) {
        const ecgCountEl = document.getElementById('glove-ecg-count');
        if (ecgCountEl) ecgCountEl.textContent = pkt.ecg_samples.length;
      }

      const axArr = pkt.ax_samples || (pkt.ax !== undefined ? [pkt.ax] : []);
      const ayArr = pkt.ay_samples || (pkt.ay !== undefined ? [pkt.ay] : []);
      const azArr = pkt.az_samples || (pkt.az !== undefined ? [pkt.az] : []);
      if (axArr.length) {
        const lastAx = Number(axArr[axArr.length - 1]);
        const lastAy = Number(ayArr[ayArr.length - 1]);
        const lastAz = Number(azArr[azArr.length - 1]);
        const gAx = document.getElementById('glove-ax');
        const gAy = document.getElementById('glove-ay');
        const gAz = document.getElementById('glove-az');
        if (gAx) gAx.textContent = lastAx.toFixed(2);
        if (gAy) gAy.textContent = lastAy.toFixed(2);
        if (gAz) gAz.textContent = lastAz.toFixed(2);

        const imuStatus = document.getElementById('glove-imu-status');
        if (imuStatus) {
          const hasTremor = Math.abs(lastAx) > 0.25 || Math.abs(lastAy) > 0.25 || pkt.simulated_disease === 'parkinsons';
          if (hasTremor) {
            imuStatus.textContent = 'Tremor (4.5 Hz)';
            imuStatus.style.color = 'var(--accent-amber)';
          } else {
            imuStatus.textContent = 'Steady';
            imuStatus.style.color = 'var(--accent-green)';
          }
        }
      }

      const gpsStatus = document.getElementById('glove-gps-status');
      const gpsSats = document.getElementById('glove-gps-sats');
      if (gpsStatus) {
        if (pkt.gps_valid) {
          gpsStatus.textContent = 'Valid Fix';
          gpsStatus.style.color = 'var(--accent-green)';
        } else {
          gpsStatus.textContent = 'Searching';
          gpsStatus.style.color = 'var(--accent-amber)';
        }
      }
      if (gpsSats) gpsSats.textContent = pkt.satellites || 0;
    }

    function updateBeltCard(pkt) {
      if (!pkt) return;
      beltPktCount++;
      const pktBadge = document.getElementById('belt-pkt-badge');
      if (pktBadge) pktBadge.textContent = `${beltPktCount} packets`;

      const syncPill = document.getElementById('belt-sync-pill');
      if (syncPill) {
        syncPill.textContent = 'Online (2 Hz)';
        syncPill.style.color = 'var(--accent-green)';
      }

      // Posture
      const posture = pkt.posture || (pkt.sensor_fusion ? pkt.sensor_fusion.activity_context : 'STANDING');
      const postureEl = document.getElementById('belt-val-posture');
      if (postureEl) postureEl.textContent = posture;

      const postureIconEl = document.getElementById('belt-posture-icon');
      if (postureIconEl) {
        const icons = {
          'STANDING': 'ðŸ§',
          'SITTING': 'ðŸª‘',
          'LYING': 'ðŸ›Œ',
          'WALKING': 'ðŸš¶',
          'RUNNING': 'ðŸƒ'
        };
        postureIconEl.textContent = icons[posture] || 'ðŸ§';
      }

      // Fall Status
      const isFall = Boolean(pkt.fall_detected);
      const fallValEl = document.getElementById('belt-val-fall');
      const fallIconEl = document.getElementById('belt-fall-icon');
      const fallSubEl = document.getElementById('belt-fall-sub');
      const badgeFallStatus = document.getElementById('badge-fall-status');

      if (isFall) {
        if (fallValEl) {
          fallValEl.textContent = 'FALL ALERT!';
          fallValEl.style.color = 'var(--accent-red)';
        }
        if (fallIconEl) fallIconEl.textContent = 'ðŸš¨';
        if (fallSubEl) {
          fallSubEl.textContent = 'Emergency latch sustained';
          fallSubEl.style.color = 'var(--accent-red)';
        }
        if (badgeFallStatus) {
          badgeFallStatus.textContent = 'FALL DETECTED (EMERGENCY)';
          badgeFallStatus.style.background = 'rgba(239, 68, 68, 0.2)';
          badgeFallStatus.style.color = 'var(--accent-red)';
          badgeFallStatus.style.borderColor = 'rgba(239, 68, 68, 0.5)';
        }
      } else {
        if (fallValEl) {
          fallValEl.textContent = 'SAFE';
          fallValEl.style.color = 'var(--accent-green)';
        }
        if (fallIconEl) fallIconEl.textContent = 'ðŸ›¡ï¸';
        if (fallSubEl) {
          fallSubEl.textContent = 'Impact & latch monitor';
          fallSubEl.style.color = 'var(--text-dim)';
        }
        if (badgeFallStatus) {
          badgeFallStatus.textContent = 'NO FALLS';
          badgeFallStatus.style.background = 'rgba(16, 185, 129, 0.15)';
          badgeFallStatus.style.color = 'var(--accent-green)';
          badgeFallStatus.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        }
      }

      // Magnitude
      const ax = pkt.ax !== undefined ? Number(pkt.ax) : 0;
      const ay = pkt.ay !== undefined ? Number(pkt.ay) : 0;
      const az = pkt.az !== undefined ? Number(pkt.az) : 9.81;
      const mag = Math.sqrt(ax * ax + ay * ay + az * az);
      const magEl = document.getElementById('belt-val-mag');
      if (magEl) magEl.textContent = mag.toFixed(2);

      // Tilt
      const tilt = pkt.tilt_degrees !== undefined ? Number(pkt.tilt_degrees) : 0;
      const tiltEl = document.getElementById('belt-val-tilt');
      if (tiltEl) tiltEl.textContent = tilt.toFixed(1);
      const tiltDescEl = document.getElementById('belt-tilt-desc');
      if (tiltDescEl) {
        if (Math.abs(tilt) > 60) tiltDescEl.textContent = 'Horizontal';
        else if (Math.abs(tilt) > 30) tiltDescEl.textContent = 'Inclined';
        else tiltDescEl.textContent = 'Upright';
      }

      // Activity Variance
      const act = pkt.activity_magnitude !== undefined ? Number(pkt.activity_magnitude) : 0;
      const actEl = document.getElementById('belt-val-act');
      if (actEl) actEl.textContent = act.toFixed(2);
      const actDescEl = document.getElementById('belt-act-desc');
      if (actDescEl) {
        if (act > 12) actDescEl.textContent = 'Running';
        else if (act > 5) actDescEl.textContent = 'Walking';
        else actDescEl.textContent = 'Resting';
      }

      // 3-Axis Accel & Gyro
      const bAx = document.getElementById('belt-ax');
      if (bAx) bAx.textContent = ax.toFixed(2);
      const bAy = document.getElementById('belt-ay');
      if (bAy) bAy.textContent = ay.toFixed(2);
      const bAz = document.getElementById('belt-az');
      if (bAz) bAz.textContent = az.toFixed(2);

      const gx = pkt.gx !== undefined ? Number(pkt.gx) : 0;
      const gy = pkt.gy !== undefined ? Number(pkt.gy) : 0;
      const gz = pkt.gz !== undefined ? Number(pkt.gz) : 0;
      const bGx = document.getElementById('belt-gx');
      if (bGx) bGx.textContent = gx.toFixed(1);
      const bGy = document.getElementById('belt-gy');
      if (bGy) bGy.textContent = gy.toFixed(1);
      const bGz = document.getElementById('belt-gz');
      if (bGz) bGz.textContent = gz.toFixed(1);
    }

    function updateRiskHero(data) {
      const riskScore = data.risk_score !== undefined ? data.risk_score : 0;
      const riskEl = document.getElementById('val-risk');
      if (riskEl) {
        riskEl.textContent = riskScore;
        riskEl.style.color = (riskScore > 60) ? 'var(--accent-red)' :
                             (riskScore > 30 ? 'var(--accent-amber)' : 'var(--accent-green)');
      }

      const bar = document.getElementById('risk-progress-bar');
      if (bar) {
        bar.style.width = Math.min(100, Math.max(0, riskScore)) + '%';
        bar.style.background = (riskScore > 60) ? 'linear-gradient(90deg, #f59e0b, #ef4444)' :
                               (riskScore > 30 ? 'linear-gradient(90deg, #06b6d4, #f59e0b)' : 'linear-gradient(90deg, #10b981, #06b6d4)');
      }

      const badge = document.getElementById('badge-risk-status');
      const label = document.getElementById('risk-level-label');
      if (riskScore > 60) {
        if (badge) {
          badge.textContent = 'HIGH EMERGENCY RISK';
          badge.style.background = 'rgba(239, 68, 68, 0.2)';
          badge.style.color = 'var(--accent-red)';
          badge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
        }
        if (label) {
          label.textContent = 'Critical Physiological Risk Detected';
          label.style.color = 'var(--accent-red)';
        }
      } else if (riskScore > 30) {
        if (badge) {
          badge.textContent = 'MODERATE ELEVATED';
          badge.style.background = 'rgba(245, 158, 11, 0.2)';
          badge.style.color = 'var(--accent-amber)';
          badge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
        }
        if (label) {
          label.textContent = 'Moderate Physiological Warning';
          label.style.color = 'var(--accent-amber)';
        }
      } else {
        if (badge) {
          badge.textContent = 'NORMAL VITALS';
          badge.style.background = 'rgba(16, 185, 129, 0.15)';
          badge.style.color = 'var(--accent-green)';
          badge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        }
        if (label) {
          label.textContent = 'Low Physiological Risk (Stable)';
          label.style.color = 'var(--accent-green)';
        }
      }

      const posture = (data.sensor_fusion && data.sensor_fusion.activity_context) ?
                      data.sensor_fusion.activity_context : (data.posture || 'STANDING');
      const postureEl = document.getElementById('val-posture');
      if (postureEl) postureEl.textContent = `Posture: ${posture}`;

      const fusionPill = document.getElementById('fusion-sync-pill');
      if (fusionPill) {
        if (data.sensor_fusion && data.sensor_fusion.waist_linked) {
          fusionPill.textContent = 'Cross-Verified (Dual-Node)';
          fusionPill.style.color = 'var(--accent-green)';
        } else {
          fusionPill.textContent = 'Single Node';
          fusionPill.style.color = 'var(--text-dim)';
        }
      }
    }

    // Attempt to load latest raw telemetry record from database on startup
    fetch('/api/raw/latest')
      .then(r => r.json())
      .then(data => {
        if (data.status === 'ok' && data.raw_packet) {
          updateRawInspector(data.raw_packet);
          updateVitalsCards(data.raw_packet);
          updateRiskHero(data.raw_packet);
          logMsg('Loaded last known raw hardware packet from database.');
        }
      })
      .catch(() => {});

    // Connect Server-Sent Events (SSE)
    const evtSource = new EventSource('/api/stream');
    const logEl = document.getElementById('event-log');

    function logMsg(txt) {
      const line = document.createElement('div');
      line.textContent = `[${new Date().toLocaleTimeString()}] ${txt}`;
      logEl.prepend(line);
    }

    evtSource.addEventListener('connected', () => {
      document.getElementById('conn-status').textContent = 'SSE Connected (Realtime)';
      logMsg('Connected to local streaming hub.');
    });

    evtSource.addEventListener('telemetry', (e) => {
      const data = JSON.parse(e.data);
      const raw = data.raw_packet || data;
      const devId = (raw.device_id || data.device_id || '').toUpperCase();
      const nodeType = (raw.node_type || data.node_type || '').toLowerCase();
      const isWaist = (nodeType === 'waist' || devId.includes('WAIST') || devId.includes('BELT'));

      logMsg(`Packet from ${data.device_id || 'node'} (${data.node_type || 'unknown'}): HR=${data.heart_rate !== undefined ? data.heart_rate : '--'}, SpO2=${data.spo2 !== undefined ? data.spo2 : '--'}, Risk=${data.risk_score || 0}`);

      if (isWaist) {
        updateBeltCard(raw);
      } else {
        updateRawInspector(raw);
        updateVitalsCards(raw);
      }

      // If packet contains waist fusion context, keep belt card synchronized
      if (data.sensor_fusion && data.sensor_fusion.waist_linked) {
        updateBeltCard({
          posture: data.sensor_fusion.activity_context,
          activity_magnitude: data.sensor_fusion.activity_magnitude,
          tilt_degrees: data.sensor_fusion.tilt_degrees,
          fall_detected: Boolean(data.fall_detected)
        });
      }

      // Update Risk Score Hero Card
      updateRiskHero(data);

      if (data.sensor_fusion && data.sensor_fusion.activity_context) {
        document.getElementById('txt-fusion').textContent = `${data.sensor_fusion.activity_context} (linked: ${data.sensor_fusion.waist_linked})`;
      }

      if (data.screenings) {
        for (const [key, scr] of Object.entries(data.screenings)) {
          const bar = document.getElementById(`bar-${key}`);
          const txt = document.getElementById(`txt-${key}`);
          if (bar && txt && scr.likelihood_percent !== undefined) {
            const pct = Math.round(scr.likelihood_percent);
            bar.style.width = pct + '%';
            txt.textContent = pct + '%';
            bar.style.background = pct > 65 ? 'var(--accent-red)' : (pct > 40 ? 'var(--accent-amber)' : 'var(--accent-cyan)');
          }
        }
      }

      // Manage emergency banner auto-clear when vitals normalize
      if (!data.emergency && (data.risk_score === undefined || data.risk_score < 60)) {
        const banner = document.getElementById('emergency-banner');
        if (banner && banner.classList.contains('emergency-active')) {
          banner.classList.remove('emergency-active');
        }
      }

      if (data.correlation) {
        document.getElementById('txt-correlation').textContent = `Score: ${data.correlation.correlation_score} (Concordance: ${data.correlation.concordance})`;
      }
      if (data.early_sepsis) {
        document.getElementById('txt-early-sepsis').textContent = `SIRS: ${data.early_sepsis.sirs_score || 0} (${data.early_sepsis.early_warning ? 'WARNING' : 'Normal'})`;
      }
      if (data.silent_hypoxia) {
        const eta = data.silent_hypoxia.eta_to_85_min;
        document.getElementById('txt-silent-hypoxia').textContent = eta ? `ETA <85%: ${eta} min` : 'Stable';
      }
      if (data.sos_gesture && data.sos_gesture.triggered) {
        document.getElementById('txt-sos').textContent = 'SOS TRIGGERED!';
        document.getElementById('txt-sos').style.color = 'var(--accent-red)';
      }
    });

    let currentEmergencyAlertId = null;

    evtSource.addEventListener('emergency', (e) => {
      const alert = JSON.parse(e.data);
      if (alert.alert && alert.alert.id) {
        currentEmergencyAlertId = alert.alert.id;
      }
      document.getElementById('emergency-banner').classList.add('emergency-active');
      document.getElementById('emergency-detail').textContent = `Device: ${alert.device_id} | Risk: ${alert.risk_score} | ${alert.alert ? alert.alert.message : 'Emergency Triggered'}`;
    });

    evtSource.addEventListener('settings_updated', (e) => {
      try {
        const data = JSON.parse(e.data);
        if (typeof data.sync !== 'undefined') {
          updateSyncUI(data.sync);
        }
      } catch (err) {}
    });

    function updateSyncUI(isSync) {
      const statusEl = document.getElementById('hdr-sync-status');
      const dotEl = document.getElementById('hdr-sync-dot');
      const btn = document.getElementById('btn-sync-toggle');
      const chk = document.getElementById('set-sync-enabled');
      if (chk) chk.checked = !!isSync;
      if (statusEl && dotEl && btn) {
        if (isSync) {
          statusEl.textContent = 'ON';
          statusEl.style.color = 'var(--accent-cyan)';
          dotEl.style.background = 'var(--accent-cyan)';
          btn.style.borderColor = 'rgba(6,182,212,0.6)';
          btn.style.boxShadow = '0 0 10px rgba(6,182,212,0.2)';
        } else {
          statusEl.textContent = 'OFF';
          statusEl.style.color = 'var(--text-dim)';
          dotEl.style.background = 'var(--text-dim)';
          btn.style.borderColor = 'rgba(255,255,255,0.1)';
          btn.style.boxShadow = 'none';
        }
      }
    }

    function toggleSyncQuick() {
      fetch('/api/settings/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      }).then(r => r.json()).then(res => {
        updateSyncUI(res.sync);
      }).catch(err => console.error('Error toggling sync:', err));
    }

    function openSettings() {
      document.getElementById('settings-msg').textContent = '';
      fetch('/api/settings').then(r => r.json()).then(s => {
        updateSyncUI(s.sync);
        document.getElementById('set-sync-enabled').checked = !!s.sync;
        document.getElementById('set-fb-enabled').checked = !!s.firebase_enabled;
        document.getElementById('set-fb-interval').value = s.firebase_interval_sec;
        document.getElementById('fb-avail-note').textContent = s.firebase_available
          ? 'Firebase bridge connected'
          : 'Firebase bridge not active on this server (no ServiceAccount.json or --no-firebase)';
        document.getElementById('settings-modal').classList.add('open');
      });
    }
    function closeSettings() {
      document.getElementById('settings-modal').classList.remove('open');
    }
    function saveSettings() {
      const body = {
        sync: document.getElementById('set-sync-enabled').checked,
        firebase_enabled: document.getElementById('set-fb-enabled').checked,
        firebase_interval_sec: parseFloat(document.getElementById('set-fb-interval').value)
      };
      if (isNaN(body.firebase_interval_sec)) body.firebase_interval_sec = 0;
      fetch('/api/settings', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
      }).then(r => r.json()).then(s => {
        updateSyncUI(s.sync);
        document.getElementById('set-fb-interval').value = s.firebase_interval_sec;
        const msg = document.getElementById('settings-msg');
        msg.textContent = 'Saved âœ“';
        setTimeout(() => { msg.textContent = ''; }, 2000);
      });
    }

    // Initial fetch of settings to initialize header sync badge
    fetch('/api/settings').then(r => r.json()).then(s => {
      updateSyncUI(s.sync);
    }).catch(() => {});

    function dismissEmergency() {
      document.getElementById('emergency-banner').classList.remove('emergency-active');
      const payload = currentEmergencyAlertId ? { alert_id: currentEmergencyAlertId } : { alert_id: 'all' };
      fetch('/api/alerts/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).catch(err => console.error('Error resolving alert:', err));
    }

    // â”€â”€ Disease Simulation Controller â”€â”€
    let simPollInterval = null;

    function updateSimulationUI(st) {
      const badge = document.getElementById('sim-status-badge');
      const label = document.getElementById('sim-status-label');
      const dot = document.getElementById('sim-status-dot');
      const stopBtn = document.getElementById('sim-stop-btn');
      const progContainer = document.getElementById('sim-progress-container');
      const progBar = document.getElementById('sim-progress-bar');
      const progText = document.getElementById('sim-progress-text');
      const progPct = document.getElementById('sim-progress-pct');

      // Reset any active simulation highlights on buttons
      document.querySelectorAll('.sim-btn').forEach(b => b.classList.remove('simulating'));

      if (st && st.active) {
        badge.className = 'sim-badge-active';
        dot.className = 'sim-dot-pulse';
        const rem = Math.max(0, Math.round(st.duration_sec - st.elapsed_sec));
        label.textContent = `SIMULATING: ${st.label.toUpperCase()} (${rem}s remaining)`;
        stopBtn.style.display = 'inline-flex';
        progContainer.style.display = 'block';

        const pct = Math.round((st.progress || 0) * 100);
        progBar.style.width = pct + '%';
        progPct.textContent = pct + '%';
        progText.textContent = `Feeding ${st.label} vectors (${st.needs}) â€¢ Physical glove bypassed`;

        const activeBtn = document.getElementById(`btn-sim-${st.disease}`);
        if (activeBtn) activeBtn.classList.add('simulating');

        if (!simPollInterval) {
          simPollInterval = setInterval(fetchSimulationStatus, 1000);
        }
      } else {
        badge.className = 'sim-badge-idle';
        dot.className = 'sim-dot-idle';
        label.textContent = 'Live Glove Mode';
        stopBtn.style.display = 'none';
        progContainer.style.display = 'none';
        if (simPollInterval) {
          clearInterval(simPollInterval);
          simPollInterval = null;
        }
      }
    }

    function fetchSimulationStatus() {
      fetch('/api/simulate/status')
        .then(r => r.json())
        .then(st => updateSimulationUI(st))
        .catch(() => {});
    }

    function startSimulation(disease) {
      fetch('/api/simulate/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ disease: disease })
      })
      .then(r => r.json())
      .then(st => {
        updateSimulationUI(st);
        logMsg(`Started disease simulation: ${st.label || disease}`);
      })
      .catch(err => console.error('Simulate start error:', err));
    }

    function stopSimulation() {
      fetch('/api/simulate/stop', { method: 'POST' })
        .then(r => r.json())
        .then(st => {
          updateSimulationUI(st);
          logMsg('Stopped disease simulation. Restored live glove telemetry.');
        })
        .catch(err => console.error('Simulate stop error:', err));
    }

    // Connect SSE listener for simulation state broadcasts
    evtSource.addEventListener('simulation_state', (e) => {
      try {
        const st = JSON.parse(e.data);
        updateSimulationUI(st);
      } catch (err) {}
    });

    // Initial check on page load
    fetchSimulationStatus();
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def dashboard_page():
    """Serve the real-time local web dashboard."""
    return render_template_string(DASHBOARD_HTML)


# ==============================================================================
#  CLI Entry Point & Runner
# ==============================================================================

def _start_udp_discovery_responder(http_port: int, local_ips: List[str]):
    """Background daemon that replies to ESP32 broadcast queries on UDP port 5005."""
    def _responder_loop():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", 5005))
            primary_ip = local_ips[0] if local_ips else "127.0.0.1"
            while True:
                data, addr = sock.recvfrom(1024)
                if b"DISCOVER_SMARTGLOVE_HUB" in data:
                    reply = f"SMARTGLOVE_HUB_IP:{primary_ip}:{http_port}".encode("utf-8")
                    sock.sendto(reply, addr)
        except Exception:
            pass

    t = threading.Thread(target=_responder_loop, daemon=True)
    t.start()


def main():
    global backend, firebase_bridge, firebase_worker_thread

    parser = argparse.ArgumentParser(description="SmartGlove Local Flask Ingestion Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    parser.add_argument("--db", default="smartwear.db", help="SQLite database path (default: smartwear.db)")
    parser.add_argument("--no-firebase", action="store_true", help="Disable background Firebase sync for mobile/web app")
    args = parser.parse_args()
    load_settings()

    print("=" * 76)
    print("  SMARTGLOVE AI HEALTHCARE HUB â€” FLASK LOCAL SERVER")
    print("=" * 76)
    print(f"  Database: {args.db}")
    print("  Initializing AI Models & Clinical Pipelines...")

    backend = SmartWearBackend(db_path=args.db)
    print("  âœ“ AI Inference Models Loaded (Arrhythmia, Parkinson's, Hypoxemia, Sepsis, Stress)")
    print("  âœ“ Advanced Clinical Engines Active (Correlation, Fusion, Sepsis, Hypoxia, SOS)")

    local_ips = get_local_ip_addresses()
    primary_ip = local_ips[0] if local_ips else "127.0.0.1"

    # Start local UDP Auto-Discovery Responder
    _start_udp_discovery_responder(args.port, local_ips)
    print("  âœ“ Local UDP Auto-Discovery Responder active on UDP port 5005")

    # Initialize Firebase bridge for App integration if credentials exist
    if not args.no_firebase and os.path.exists("ServiceAccount.json"):
        try:
            from firebase_bridge import FirebaseTelemetryBridge
            firebase_bridge = FirebaseTelemetryBridge(backend)
            firebase_worker_thread = threading.Thread(target=_firebase_sync_worker, daemon=True)
            firebase_worker_thread.start()

            # Publish dynamic IP configuration to Firebase for ESP32 auto-discovery
            try:
                firebase_bridge.get_reference("hub/config").set({
                    "local_ip": primary_ip,
                    "port": args.port,
                    "endpoint": f"http://{primary_ip}:{args.port}/api/telemetry",
                    "status": "online",
                    "updated_at": time.time(),
                })
                print(f"  âœ“ Dynamic Hub IP published to Firebase (/hub/config) -> {primary_ip}:{args.port}")
            except Exception as e:
                print(f"  ! Could not write hub config to Firebase: {e}")

            # Start real-time listener and polling for mobile app facial photo requests
            firebase_bridge.start_facial_stress_listener()

            def _facial_poll_worker():
                while True:
                    try:
                        if firebase_bridge:
                            firebase_bridge.poll_facial_stress_once()
                    except Exception:
                        pass
                    time.sleep(2.0)

            threading.Thread(target=_facial_poll_worker, daemon=True).start()
            print("  âœ“ Firebase Realtime Database Bridge active (Syncing to App at /processed/ and /alerts/)")
            print("  âœ“ Facial Expression Stress Poller active (Listening on /facial_stress/request)")
        except Exception as exc:
            print(f"  ! Firebase bridge disabled: {exc}")

    print("\n  Network Ingestion Endpoints (Auto-Discovered by ESP32s):")
    for ip in local_ips:
        print(f"    â€¢ Local LAN URL:       http://{ip}:{args.port}/api/telemetry")
        print(f"    â€¢ Live Dashboard UI:   http://{ip}:{args.port}/")
    print("=" * 76)
    print(f"  Starting Flask server on http://{args.host}:{args.port} ...\n")

    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()

