from __future__ import annotations

import json
from urllib import error, request


BASE_URL = "http://127.0.0.1:8000/api/v1"


def fetch(path: str) -> tuple[int, str]:
    method = "POST" if path == "/portfolio/refresh" else "GET"
    req = request.Request(
        f"{BASE_URL}{path}",
        method=method,
        headers={"X-Tenant-ID": "family-office-demo"},
    )
    with request.urlopen(req) as response:
        return response.status, response.read().decode("utf-8")


def main() -> None:
    endpoints = [
        "/health",
        "/terminal/snapshot",
        "/portfolio/refresh",
        "/portfolio",
        "/portfolio/performance",
        "/portfolio/history",
        "/portfolio/barbell",
        "/orders",
        "/orders/fills",
        "/risk/portfolio",
        "/risk/positions",
        "/risk/scenarios",
        "/risk/correlations",
        "/regime",
        "/signals/AAPL",
        "/forecast/AAPL",
        "/research/screener",
        "/research/factors",
        "/research/sectors",
        "/admin/users",
        "/admin/audit",
        "/admin/events",
        "/admin/events/summary",
    ]

    for path in endpoints:
        status, body = fetch(path)
        payload = json.loads(body)
        print(f"{path} -> {status}")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    try:
        main()
    except error.URLError as exc:
        print(f"Smoke test failed: {exc}")
