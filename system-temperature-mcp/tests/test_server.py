from __future__ import annotations

import io
import json
from datetime import timedelta

from system_temperature_mcp import server


def test_japan_timezone_falls_back_to_fixed_jst(monkeypatch) -> None:
    def missing_zoneinfo(_key: str):
        raise server.ZoneInfoNotFoundError

    monkeypatch.setattr(server, "ZoneInfo", missing_zoneinfo)

    timezone = server._japan_timezone()

    assert timezone.utcoffset(None) == timedelta(hours=9)
    assert timezone.tzname(None) == "JST"


def test_lhm_webserver_extracts_temperatures(monkeypatch) -> None:
    payload = {
        "Text": "root",
        "Children": [
            {"Text": "CPU Package", "Value": "52.5 °C", "Children": []},
            {"Text": "SSD", "Value": "41,0 °C", "Children": []},
            {"Text": "Distance to TjMax", "Value": "47.5 °C", "Children": []},
        ],
    }
    response = io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(server.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    temperatures = server._get_lhm_webserver_temps()

    assert temperatures == [
        {
            "source": "lhm_webserver",
            "name": "CPU Package",
            "temperature_celsius": 52.5,
        },
        {
            "source": "lhm_webserver",
            "name": "SSD",
            "temperature_celsius": 41.0,
        },
    ]


def _lhm_payload_with_configuration_values() -> dict:
    """A trimmed LHM tree captured from a real machine.

    Source: Intel Core Ultra 7 155H / SK Hynix DDR5 / KIOXIA NVMe, LHM 0.9.6.
    Alongside real readings, LHM reports hardware *configuration* values in °C:
    DDR5 SPD exposes thermal limits and the sensor's own resolution, and NVMe
    exposes SMART warning/critical thresholds. Older hardware does not report
    these, so this only shows up on newer machines.
    """
    return {
        "Text": "Sensor",
        "Children": [
            {
                "Text": "Intel Core Ultra 7 155H",
                "Children": [
                    {
                        "Text": "Temperatures",
                        "Children": [
                            {"Text": "CPU Package", "Value": "47.0 °C", "Children": []},
                            {"Text": "P-Core #1", "Value": "44.0 °C", "Children": []},
                            {
                                "Text": "P-Core #1 Distance to TjMax",
                                "Value": "53.0 °C",
                                "Children": [],
                            },
                        ],
                    }
                ],
            },
            {
                "Text": "SK Hynix - HMCG78AGBSA092N (#0)",
                "Children": [
                    {
                        "Text": "Temperatures",
                        "Children": [
                            {"Text": "DIMM #0", "Value": "39.0 °C", "Children": []},
                            {
                                "Text": "Temperature Sensor Resolution",
                                "Value": "0.3 °C",
                                "Children": [],
                            },
                            {
                                "Text": "Thermal Sensor Low Limit",
                                "Value": "0.0 °C",
                                "Children": [],
                            },
                            {
                                "Text": "Thermal Sensor High Limit",
                                "Value": "55.0 °C",
                                "Children": [],
                            },
                            {
                                "Text": "Thermal Sensor Critical Low Limit",
                                "Value": "0.0 °C",
                                "Children": [],
                            },
                            {
                                "Text": "Thermal Sensor Critical High Limit",
                                "Value": "85.0 °C",
                                "Children": [],
                            },
                        ],
                    }
                ],
            },
            {
                "Text": "KBG6AZNV512G LA KIOXIA",
                "Children": [
                    {
                        "Text": "Temperatures",
                        "Children": [
                            {
                                "Text": "Composite Temperature",
                                "Value": "35.0 °C",
                                "Children": [],
                            },
                            {
                                "Text": "Warning Temperature",
                                "Value": "82.0 °C",
                                "Children": [],
                            },
                            {
                                "Text": "Critical Temperature",
                                "Value": "84.0 °C",
                                "Children": [],
                            },
                        ],
                    }
                ],
            },
        ],
    }


def test_lhm_webserver_skips_configuration_values(monkeypatch) -> None:
    """Only live readings are collected; limits and specs are not temperatures."""
    response = io.BytesIO(json.dumps(_lhm_payload_with_configuration_values()).encode())
    monkeypatch.setattr(server.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    temperatures = server._get_lhm_webserver_temps()

    assert [t["name"] for t in temperatures] == [
        "CPU Package",
        "P-Core #1",
        "DIMM #0",
        "Composite Temperature",
    ]
    # The DIMM's 0.0 °C low limit would otherwise become the minimum, and its
    # 85.0 °C critical limit the maximum.
    values = [t["temperature_celsius"] for t in temperatures]
    assert (min(values), max(values)) == (35.0, 47.0)


def test_configuration_values_do_not_change_the_reported_feeling(monkeypatch) -> None:
    """Regression: a DDR5 critical limit must not be felt as body heat.

    Before configuration values were filtered out, a DIMM's 85.0 °C critical
    limit became max_temp, so the server reported severe heat while the CPU was
    idling in the 40s. Asserting against the readings-only verdict keeps this
    independent of how the feelings themselves are worded.
    """
    response = io.BytesIO(json.dumps(_lhm_payload_with_configuration_values()).encode())
    monkeypatch.setattr(server.urllib.request, "urlopen", lambda *_args, **_kwargs: response)

    from_lhm = server.interpret_temperature(server._get_lhm_webserver_temps())
    readings_only = server.interpret_temperature([
        {"source": "lhm_webserver", "name": "CPU Package", "temperature_celsius": 47.0},
        {"source": "lhm_webserver", "name": "P-Core #1", "temperature_celsius": 44.0},
        {"source": "lhm_webserver", "name": "DIMM #0", "temperature_celsius": 39.0},
        {
            "source": "lhm_webserver",
            "name": "Composite Temperature",
            "temperature_celsius": 35.0,
        },
    ])

    assert from_lhm == readings_only


def test_windows_temperature_prefers_lhm_webserver(monkeypatch) -> None:
    expected = [
        {
            "source": "lhm_webserver",
            "name": "CPU Package",
            "temperature_celsius": 52.5,
        }
    ]
    monkeypatch.setattr(server.sys, "platform", "win32")
    monkeypatch.setattr(server, "_get_lhm_webserver_temps", lambda: expected)
    monkeypatch.setattr(
        server,
        "_get_hardware_monitor_temps",
        lambda: (_ for _ in ()).throw(AssertionError("WMI fallback should not run")),
    )

    assert server.get_windows_temperatures() == expected
