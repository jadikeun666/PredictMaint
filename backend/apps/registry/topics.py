"""
Konvensi topic MQTT — mqtt-architecture.md §Topic Structure:
  factory/{machine_slug}/{bearing_position_label}/vibration
machine_slug: lowercase, hyphenated, diturunkan dari machines.name
("CWRU Test Rig 1" -> "cwru-test-rig-1", sama persis dengan simulator).
"""
import re


def slugify_machine_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def build_vibration_topic(machine_name: str, position_label: str) -> str:
    return f"factory/{slugify_machine_name(machine_name)}/{position_label}/vibration"
