"""
esp32_bridge.py – HYGEIA Hardware Bridge
=========================================
Polls the ESP32 HYGEIA Glove (AP mode) at http://192.168.4.1/api/data
every 3 seconds and forwards valid readings to the local HYGEIA AI
server at http://localhost:5000/api/reading.

Usage (called automatically by start_demo.bat):
    python hardware/esp32_bridge.py

Prerequisites:
  - Laptop must be connected to the "HYGEIA-Monitor" WiFi hotspot
  - HYGEIA AI server (server.py) must already be running on port 5000
"""

import json
import sys
import time
import urllib.error
import urllib.request

ESP32_URL   = "http://192.168.4.1/api/data"   # ESP32 AP JSON endpoint
SERVER_URL  = "http://localhost:5000/api/reading"
POLL_EVERY  = 3       # seconds between polls
CONNECT_TIMEOUT = 4   # seconds before giving up on a single request

# ── ANSI colours (work in Windows Terminal / PowerShell 7+) ─────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def fetch_esp32():
    """GET /api/data from the ESP32. Returns parsed dict or None on error."""
    try:
        with urllib.request.urlopen(ESP32_URL, timeout=CONNECT_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.URLError as e:
        return None   # Not connected to AP yet
    except Exception as e:
        print(f"{YELLOW}[BRIDGE] ESP32 parse error: {e}{RESET}")
        return None


def post_to_server(temp, spo2, pulse):
    """POST a reading to the local HYGEIA Flask server. Returns True on success."""
    payload = json.dumps({"temp": temp, "spo2": spo2, "pulse": pulse}).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json", "X-Device": "HYGEIA-ESP32-BRIDGE"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return True, result.get("classification", "?"), result.get("confidence", 0)
    except Exception as e:
        return False, str(e), 0


def wait_for_server(max_wait=60):
    """Block until the Flask server is up, then return True."""
    print(f"{CYAN}[BRIDGE] Waiting for HYGEIA server on localhost:5000...{RESET}")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:5000/api/status", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def wait_for_esp32(max_wait=120):
    """Block until the ESP32 AP is reachable, then return True."""
    print(f"{CYAN}[BRIDGE] Waiting for ESP32 at {ESP32_URL}...{RESET}")
    print(f"         Make sure your laptop WiFi is set to {BOLD}HYGEIA-Monitor{RESET}")
    deadline = time.time() + max_wait
    dots = 0
    while time.time() < deadline:
        data = fetch_esp32()
        if data is not None:
            return True
        sys.stdout.write(".")
        sys.stdout.flush()
        dots += 1
        if dots % 20 == 0:
            print()
        time.sleep(2)
    print()
    return False


def main():
    print()
    print(f"{BOLD}{CYAN}{'='*52}{RESET}")
    print(f"{BOLD}{CYAN}  HYGEIA Hardware Bridge – ESP32 → AI Server{RESET}")
    print(f"{BOLD}{CYAN}{'='*52}{RESET}")
    print()

    # ── Step 1: Confirm Flask server is up ──────────────────────────
    if not wait_for_server(max_wait=60):
        print(f"{RED}[BRIDGE] ERROR: HYGEIA server did not start in time.{RESET}")
        sys.exit(1)
    print(f"{GREEN}[BRIDGE] HYGEIA server is ready.{RESET}")
    print()

    # ── Step 2: Wait for ESP32 AP connection ────────────────────────
    if not wait_for_esp32(max_wait=120):
        print(f"\n{RED}[BRIDGE] ERROR: Could not reach ESP32 at {ESP32_URL}{RESET}")
        print(f"         Check that your laptop WiFi shows 'HYGEIA-Monitor'.")
        sys.exit(1)
    print(f"\n{GREEN}[BRIDGE] ESP32 found! Live data bridge is now active.{RESET}")
    print(f"         Dashboard → {BOLD}http://localhost:5000{RESET}")
    print(f"         Press Ctrl+C to stop.")
    print()

    # ── Step 3: Poll → forward loop ─────────────────────────────────
    skipped   = 0
    forwarded = 0

    while True:
        data = fetch_esp32()

        if data is None:
            skipped += 1
            if skipped % 5 == 1:
                print(f"{YELLOW}[BRIDGE] ESP32 unreachable – is the glove still powered?{RESET}")
            time.sleep(POLL_EVERY)
            continue

        skipped = 0  # Reset on successful fetch

        if not data.get("valid", False):
            finger = data.get("finger", False)
            if not finger:
                print(f"{YELLOW}[BRIDGE] Waiting for finger on sensor...{RESET}")
            else:
                print(f"{YELLOW}[BRIDGE] Sensor calculating – finger detected, hold still.{RESET}")
            time.sleep(POLL_EVERY)
            continue

        temp  = data["temp"]
        spo2  = data["spo2"]
        pulse = data["pulse"]

        ok, classification, confidence = post_to_server(temp, spo2, pulse)
        if ok:
            forwarded += 1
            conf_pct = int(confidence * 100) if isinstance(confidence, float) else confidence
            status_colour = GREEN if classification == "Normal" else (
                YELLOW if classification == "Warning" else RED
            )
            print(
                f"{GREEN}[BRIDGE]{RESET} "
                f"Temp:{BOLD}{temp:.1f}°C{RESET}  "
                f"SpO2:{BOLD}{int(spo2)}%{RESET}  "
                f"BPM:{BOLD}{int(pulse)}{RESET}  →  "
                f"{status_colour}{BOLD}{classification}{RESET} "
                f"({conf_pct}% conf)"
            )
        else:
            print(f"{RED}[BRIDGE] Failed to post to server: {classification}{RESET}")

        time.sleep(POLL_EVERY)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{CYAN}[BRIDGE] Stopped. Goodbye!{RESET}")
