from __future__ import annotations

from collections import defaultdict
from threading import Lock


class InMemoryMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._request_totals: dict[tuple[str, str, int], int] = defaultdict(int)
        self._request_latency_seconds: dict[tuple[str, str], float] = defaultdict(float)

    def record_request(self, method: str, path: str, status_code: int, duration_seconds: float) -> None:
        key = (method.upper(), path, int(status_code))
        latency_key = (method.upper(), path)
        with self._lock:
            self._request_totals[key] += 1
            self._request_latency_seconds[latency_key] += duration_seconds

    def render_prometheus(self) -> str:
        lines = [
            "# HELP institutional_api_requests_total Total HTTP requests handled by the API.",
            "# TYPE institutional_api_requests_total counter",
        ]
        for (method, path, status_code), total in sorted(self._request_totals.items()):
            lines.append(
                'institutional_api_requests_total{method="%s",path="%s",status="%s"} %s'
                % (method, path, status_code, total)
            )

        lines.extend(
            [
                "# HELP institutional_api_request_latency_seconds_total Cumulative request latency in seconds.",
                "# TYPE institutional_api_request_latency_seconds_total counter",
            ]
        )
        for (method, path), total_seconds in sorted(self._request_latency_seconds.items()):
            lines.append(
                'institutional_api_request_latency_seconds_total{method="%s",path="%s"} %.6f'
                % (method, path, total_seconds)
            )

        return "\n".join(lines) + "\n"


metrics_registry = InMemoryMetrics()
