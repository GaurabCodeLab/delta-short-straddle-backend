from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    api_key: str
    api_secret: str
    base_url: str
    poll_interval_seconds: float
    max_single_order_qty: float
    max_total_option_notional: float
    log_level: str
    ssl_verify: bool


def load_settings() -> Settings:
    api_key = os.getenv("DELTA_API_KEY", "")
    api_secret = os.getenv("DELTA_API_SECRET", "")
    if not api_key or not api_secret:
        raise ValueError("Missing DELTA_API_KEY or DELTA_API_SECRET in environment.")

    return Settings(
        api_key=api_key,
        api_secret=api_secret,
        base_url=os.getenv("DELTA_BASE_URL", "https://cdn-ind.testnet.deltaex.org"),
        poll_interval_seconds=float(os.getenv("POLL_INTERVAL_SECONDS", "2")),
        max_single_order_qty=float(os.getenv("MAX_SINGLE_ORDER_QTY", "10")),
        max_total_option_notional=float(os.getenv("MAX_TOTAL_OPTION_NOTIONAL", "100000")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        ssl_verify=os.getenv("DELTA_SSL_VERIFY", "false").strip().lower() in {"1", "true", "yes", "y"},
    )
