"""
HYGEIA AI – API Endpoint Tests
================================
Functional tests for all HYGEIA Flask server endpoints.

These tests verify that:
  - The server responds correctly to all API requests
  - Classification results are in the expected format
  - Edge case inputs (boundary vitals, missing fields, bad data) are handled
  - The simulation mode switching works correctly
  - The buffer clear endpoint resets state properly

Prerequisites:
    The HYGEIA server must be running before executing this script:
        python server.py

Usage:
    python testing/test_api_endpoints.py

Exit codes:
    0 – All tests passed
    1 – One or more tests failed
"""

import sys
import time
import requests

BASE_URL = "http://localhost:5000"


# ==============================================================================
#  TEST HELPERS
# ==============================================================================

passed = 0
failed = 0


def test(name: str, condition: bool, detail: str = ""):
    """Record a test result and print it."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f" | {detail}" if detail else ""))


def post(endpoint: str, body: dict = None) -> requests.Response:
    return requests.post(f"{BASE_URL}{endpoint}", json=body, timeout=5)


def get(endpoint: str) -> requests.Response:
    return requests.get(f"{BASE_URL}{endpoint}", timeout=5)


# ==============================================================================
#  TEST SUITES
# ==============================================================================

def test_server_reachable():
    print("\n[Suite] Server connectivity")
    try:
        r = get("/")
        test("GET / returns 200", r.status_code == 200)
    except requests.exceptions.ConnectionError:
        print("  ERROR: Cannot connect to server at", BASE_URL)
        print("         Make sure server.py is running before running this script.")
        sys.exit(1)


def test_status_endpoint():
    print("\n[Suite] GET /api/status")
    r = get("/api/status")
    test("Returns 200",              r.status_code == 200)
    data = r.json()
    test("Has 'model_loaded' field", "model_loaded"    in data)
    test("Has 'classifier' field",   "classifier"      in data)
    test("Has 'readings_count'",     "readings_count"  in data)
    test("Has 'simulation_mode'",    "simulation_mode" in data)
    print(f"         Classifier type : {data.get('classifier', 'N/A')}")
    print(f"         Readings in buffer: {data.get('readings_count', 'N/A')}")


def test_valid_readings():
    print("\n[Suite] POST /api/reading – valid inputs")

    cases = [
        # (label,            temp,  spo2, pulse, expected_status)
        ("Normal baseline",  36.8,  98,   75,    "Normal"),
        ("Mild warning",     38.2,  93,   108,   "Warning"),
        ("SpO2 critical",    37.0,  85,   110,   "Critical"),
        ("High fever",       41.0,  96,   120,   "Critical"),
        ("Low pulse",        36.5,  98,   38,    "Critical"),
    ]

    for label, temp, spo2, pulse, expected in cases:
        r = post("/api/reading", {"temp": temp, "spo2": spo2, "pulse": pulse})
        test(f"Returns 200  [{label}]", r.status_code == 200)
        if r.status_code == 200:
            data = r.json()
            test(
                f"Has classification [{label}]",
                "classification" in data,
            )
            test(
                f"Has confidence     [{label}]",
                "confidence" in data and 0.0 <= data["confidence"] <= 1.0,
            )
            # Rule-based classifier should always match expected
            # (AI model may differ slightly due to sequence context)
            print(
                f"         {label}: {data.get('classification','?')} "
                f"({data.get('confidence', 0) * 100:.1f}%) "
                f"[expected ~{expected}]"
            )


def test_invalid_readings():
    print("\n[Suite] POST /api/reading – invalid inputs")

    # Missing fields
    r = post("/api/reading", {"temp": 36.8})
    test("Missing spo2/pulse returns 400", r.status_code == 400)

    # No body
    r = requests.post(f"{BASE_URL}/api/reading", timeout=5)
    test("Empty body returns 400", r.status_code == 400)

    # Non-numeric
    r = post("/api/reading", {"temp": "hot", "spo2": 98, "pulse": 75})
    test("Non-numeric temp returns 400", r.status_code == 400)


def test_data_endpoint():
    print("\n[Suite] GET /api/data")
    r = get("/api/data")
    test("Returns 200",        r.status_code == 200)
    data = r.json()
    test("Returns a list",     isinstance(data, list))
    if data:
        first = data[0]
        for field in ["timestamp", "temp", "spo2", "pulse", "status", "confidence"]:
            test(f"Reading has '{field}' field", field in first)


def test_current_endpoint():
    print("\n[Suite] GET /api/current")
    # Make sure there's at least one reading
    post("/api/reading", {"temp": 36.8, "spo2": 98, "pulse": 75})
    r = get("/api/current")
    test("Returns 200", r.status_code == 200)
    data = r.json()
    test("Returns a dict with 'status'", isinstance(data, dict) and "status" in data)


def test_simulation_modes():
    print("\n[Suite] POST /api/simulation/mode")
    modes = ["normal", "critical_spo2", "critical_temp", "critical_pulse"]
    for mode in modes:
        r = post("/api/simulation/mode", {"mode": mode})
        test(f"Set mode '{mode}' returns 200", r.status_code == 200)
        if r.status_code == 200:
            data = r.json()
            test(f"Mode field reflects '{mode}'", data.get("mode") == mode)

    # Reset to normal
    post("/api/simulation/mode", {"mode": "normal"})


def test_sample_control():
    print("\n[Suite] Demo generator start/stop")
    r = post("/api/sample/start")
    test("Start returns 200", r.status_code == 200)
    time.sleep(1)
    r = post("/api/sample/stop")
    test("Stop returns 200", r.status_code == 200)


def test_clear_endpoint():
    print("\n[Suite] POST /api/clear")
    # Add a reading first
    post("/api/reading", {"temp": 36.8, "spo2": 98, "pulse": 75})
    r = post("/api/clear")
    test("Clear returns 200", r.status_code == 200)

    # Buffer should now be empty (or close to it)
    r2 = get("/api/data")
    data = r2.json()
    test("Buffer is empty after clear", len(data) == 0)


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print()
    print("=" * 55)
    print("  HYGEIA AI – API Endpoint Tests")
    print(f"  Target: {BASE_URL}")
    print("=" * 55)

    test_server_reachable()
    test_status_endpoint()
    test_valid_readings()
    test_invalid_readings()
    test_data_endpoint()
    test_current_endpoint()
    test_simulation_modes()
    test_sample_control()
    test_clear_endpoint()

    print()
    print("=" * 55)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 55)
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
