"""
HYGEIA AI – Rule-Based Classifier Unit Tests
=============================================
Unit tests for the rule-based fallback classification logic.

These tests verify the classification boundaries for all three health status
classes (Normal, Warning, Critical) without requiring the server to be running
or TensorFlow to be installed.

The rule-based classifier is defined inline here (mirroring server.py) so
these tests remain standalone and fast.

Usage:
    python testing/test_classification.py

All test cases are documented with their expected clinical reasoning.
"""

import sys


# ==============================================================================
#  INLINE CLASSIFIER (mirrors server.py rule_based_classify)
# ==============================================================================

STATUS_NAMES = {0: "Normal", 1: "Warning", 2: "Critical"}


def rule_based_classify(temp: float, spo2: float, pulse: float):
    """
    Deterministic rule-based classifier.

    Critical thresholds (any one triggers Critical):
        SpO2  < 90%     – severe hypoxaemia
        Temp  > 40.0 C  – high fever / hyperthermia
        Temp  < 35.0 C  – hypothermia
        Pulse > 150 BPM – severe tachycardia
        Pulse < 40  BPM – severe bradycardia

    Warning thresholds (any one triggers Warning, if not Critical):
        SpO2  < 95%     – mild hypoxaemia
        Temp  > 38.5 C  – low-grade fever
        Temp  < 36.0 C  – mild hypothermia
        Pulse > 120 BPM – tachycardia
        Pulse < 55  BPM – bradycardia

    Returns:
        (status_index: int, status_name: str)
    """
    if spo2 < 90 or temp > 40.0 or temp < 35.0 or pulse > 150 or pulse < 40:
        return 2, STATUS_NAMES[2]   # Critical

    if spo2 < 95 or temp > 38.5 or temp < 36.0 or pulse > 120 or pulse < 55:
        return 1, STATUS_NAMES[1]   # Warning

    return 0, STATUS_NAMES[0]       # Normal


# ==============================================================================
#  TEST HELPERS
# ==============================================================================

passed = 0
failed = 0


def expect(label: str, temp: float, spo2: float, pulse: float, expected_idx: int):
    """
    Assert that the classifier returns the expected class for given vitals.
    Prints PASS or FAIL with the actual vs expected result.
    """
    global passed, failed
    idx, name = rule_based_classify(temp, spo2, pulse)
    expected_name = STATUS_NAMES[expected_idx]
    ok = (idx == expected_idx)

    if ok:
        passed += 1
        print(f"  PASS  [{name:8s}] {label}")
    else:
        failed += 1
        print(f"  FAIL  [{name:8s}] {label}")
        print(f"         Expected: {expected_name} | Got: {name}")
        print(f"         Input: Temp={temp}, SpO2={spo2}, Pulse={pulse}")


# ==============================================================================
#  TEST CASES
# ==============================================================================

def test_normal_cases():
    print("\n[Suite] Normal classification")
    # All vitals squarely within normal range
    expect("Healthy adult at rest",        37.0, 98,  75, 0)
    expect("Slightly elevated pulse",      36.8, 97,  90, 0)
    expect("Lower end of normal temp",     36.2, 99,  65, 0)
    expect("Upper end of normal temp",     38.4, 96, 118, 0)
    expect("Normal with high SpO2",        36.5, 100, 60, 0)
    expect("Boundary: SpO2 exactly 95",    37.0, 95,  75, 0)
    expect("Boundary: Pulse exactly 120",  37.0, 97, 120, 0)
    expect("Boundary: Pulse exactly 55",   37.0, 97,  55, 0)
    expect("Boundary: Temp exactly 36.0",  36.0, 97,  75, 0)
    expect("Boundary: Temp exactly 38.5",  38.5, 97,  75, 0)


def test_warning_cases():
    print("\n[Suite] Warning classification")
    # One or more vitals just outside normal range but not critical
    expect("Low SpO2 at 93%",              37.0, 93,  75, 1)
    expect("SpO2 just below 95",           37.0, 94,  75, 1)
    expect("Mild fever at 38.7",           38.7, 97,  75, 1)
    expect("Mild hypothermia at 35.8",     35.8, 97,  75, 1)
    expect("Elevated pulse at 125",        37.0, 97, 125, 1)
    expect("Low pulse at 50",              37.0, 97,  50, 1)
    expect("Combined mild warning",        38.3, 94, 115, 1)
    expect("Warning: low temp + high BPM", 35.5, 97, 118, 1)


def test_critical_cases():
    print("\n[Suite] Critical classification")
    # Any vital in the critical range should override to Critical
    expect("Severe hypoxaemia SpO2=85",    37.0, 85,  75, 2)
    expect("SpO2 just below 90",           37.0, 89,  75, 2)
    expect("High fever 41.0",             41.0, 97,  85, 2)
    expect("Temp just above 40.0",        40.1, 97,  85, 2)
    expect("Hypothermia 34.5",            34.5, 97,  75, 2)
    expect("Temp just below 35.0",        34.9, 97,  75, 2)
    expect("Tachycardia pulse=155",        37.0, 97, 155, 2)
    expect("Pulse just above 150",         37.0, 97, 151, 2)
    expect("Bradycardia pulse=35",         37.0, 97,  35, 2)
    expect("Pulse just below 40",          37.0, 97,  39, 2)
    expect("Multiple critical indicators", 41.0, 82, 160, 2)


def test_critical_overrides_warning():
    print("\n[Suite] Critical overrides Warning")
    # Even if a warning-level vital is present, a critical one takes precedence
    expect("Critical SpO2 + warning temp",   39.0, 85, 100, 2)
    expect("Critical pulse + warning SpO2",  37.0, 93, 160, 2)
    expect("Critical temp + warning pulse",  41.0, 97, 125, 2)


def test_edge_cases():
    print("\n[Suite] Edge / boundary cases")
    expect("SpO2 = 90 (boundary, not critical)",  37.0, 90,  75, 1)  # 90 is NOT < 90
    expect("Pulse = 150 (boundary, not critical)", 37.0, 97, 150, 1)  # 150 is NOT > 150
    expect("Temp = 40.0 (boundary, not critical)", 40.0, 97,  75, 1)  # 40.0 is NOT > 40.0
    expect("Temp = 35.0 (boundary, not critical)", 35.0, 97,  75, 1)  # 35.0 is NOT < 35.0
    expect("All vitals at normal maximums",        38.5, 95, 120, 0)
    expect("All vitals at normal minimums",        36.0, 95,  55, 0)


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    print()
    print("=" * 55)
    print("  HYGEIA AI – Rule-Based Classifier Unit Tests")
    print("=" * 55)

    test_normal_cases()
    test_warning_cases()
    test_critical_cases()
    test_critical_overrides_warning()
    test_edge_cases()

    print()
    print("=" * 55)
    print(f"  Results: {passed} passed, {failed} failed  ({passed + failed} total)")
    print("=" * 55)
    print()

    if failed == 0:
        print("  All classification rules are correct.")
    else:
        print(f"  {failed} test(s) failed – review the rules in server.py.")
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
