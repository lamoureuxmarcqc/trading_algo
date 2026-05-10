from __future__ import annotations

import json
from urllib import error, request


BASE_URL = "http://127.0.0.1:8000/api/v1"


def fetch(path: str) -> tuple[int, str]:
    method = "POST" if path == "/portfolio/refresh" else "GET"
    with request.urlopen(request.Request(f"{BASE_URL}{path}", method=method)) as response:
        return response.status, response.read().decode("utf-8")


def main() -> None:
    endpoints = [
        "/health",
        "/portfolio/refresh",
        "/portfolio",
        "/portfolio/performance",
        "/portfolio/history",
        "/orders",
        "/orders/fills",
        "/risk/portfolio",
        "/risk/positions",
        "/regime",
        "/signals/AAPL",
        "/forecast/AAPL",
        "/research/screener",
        "/research/factors",
        "/research/sectors",
        "/admin/users",
        "/admin/audit",
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
