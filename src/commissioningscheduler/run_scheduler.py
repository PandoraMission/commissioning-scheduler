# example_usage.py
"""
Example usage of the commissioning scheduler.
"""

from datetime import datetime
import logging
from commissioningscheduler import (
    schedule_observations,
    analyze_schedule_diagnostics,
)

import xml.etree.ElementTree as ET
import os
import re

from astropy.time import Time

# Configure logging for verbose output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configuration
xml_dir = "../../docs/xml_in_v5_01122026/"
output_path = "./master_schedule.xml"
cvz_coords = (120.0, 8.5)

# TLE (Two-Line Element set for satellite orbit)
# tle1 = "1 99152U 80229J   26014.83069444  .00000000  00000-0  37770-3 0    08"
# tle2 = "2 99152  97.8005  15.9441 0003693 273.7015  26.6009 14.87738574    04"
tle1 = "1 99152U 80229J   26017.75114583  .00000000  00000-0  37770-3 0    07"
tle2 = "2 99152  97.8003  18.8171 0003829 259.1072 192.9926 14.87777498    09"


# Commissioning period
commissioning_start = datetime(2026, 1, 21, 8, 0)
commissioning_end = datetime(2026, 2, 11, 0, 0)

# Run scheduler
print("\n" + "=" * 80)
print("COMMISSIONING SCHEDULER")
print("=" * 80 + "\n")

result = schedule_observations(
    xml_dir=xml_dir,
    output_path=output_path,
    tle_line1=tle1,
    tle_line2=tle2,
    commissioning_start=commissioning_start,
    commissioning_end=commissioning_end,
    cvz_coords=cvz_coords,
    constraints_json="constraints.json",
    max_data_volume_gb=150.0,
    verify_cvz_visibility=False,  # Set to False for faster debugging
    enable_gap_filling=True,  # Try to schedule science in gaps
)

print(f"\n{'='*80}")
print("SCHEDULING SUMMARY")
print("=" * 80)
print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"Visits: {len(result.visits)}")
print(f"Sequences: {result.total_sequences}")
print(f"Total Duration: {result.total_duration_minutes/60:.1f} hours")
print(f"Science Time: {result.total_science_minutes/60:.1f} hours")
print(f"Efficiency: {result.scheduling_efficiency:.1%}")
print(f"Data Volume: {result.total_data_volume_gb:.2f} GB")

if result.warnings:
    print(f"\nWarnings:")
    for warning in result.warnings:
        print(f"  - {warning}")

if result.unscheduled_observations:
    print(f"\nUnscheduled observations:")
    for obs in result.unscheduled_observations:
        print(f"  - {obs.obs_id}: {obs.target_name}")

print("=" * 80 + "\n")

# Run diagnostics
print("\n" + "=" * 80)
print("RUNNING DIAGNOSTICS")
print("=" * 80 + "\n")

diagnostics = analyze_schedule_diagnostics(
    sequences=result.scheduled_sequences,
    input_xml_dir=xml_dir,
    dependency_json="constraints.json",
    print_report=True,
    save_report="schedule_diagnostics.txt",
    detailed=True,
)

print(f"\nDiagnostics Summary:")
print(
    f"  Validation Errors: {sum(1 for i in diagnostics.validation_issues if i.severity == 'error')}"
)
print(
    f"  Validation Warnings: {sum(1 for i in diagnostics.validation_issues if i.severity == 'warning')}"
)
print(
    f"  Overall Status: {'✓ PASS' if not diagnostics.has_errors else '✗ FAIL'}"
)
