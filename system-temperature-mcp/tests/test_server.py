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
