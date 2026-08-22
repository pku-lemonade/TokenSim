from __future__ import annotations

from .base import BaseKVConnector


class NoopConnector(BaseKVConnector):
    name = "NoopConnector"
