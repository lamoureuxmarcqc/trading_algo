import time
import threading


class StateStore:
    """
    Source unique de vérité (remplace dcc.Store).
    """

    def __init__(self):
        self._store = {}
        self._timestamps = {}
        self._lock = threading.Lock()

    def set(self, key, value):
        with self._lock:
            self._store[key] = value
            self._timestamps[key] = time.time()

    def get(self, key):
        return self._store.get(key)

    def is_stale(self, key, ttl=60):
        if key not in self._timestamps:
            return True
        return (time.time() - self._timestamps[key]) > ttl