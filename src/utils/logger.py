"""logger.py — structured logging berformat JSON.

Kenapa tidak pakai ``print()``?

* ``print`` tidak punya waktu, tidak punya level, tidak bisa disaring.
* Begitu aplikasi masuk container, ``print`` cuma jadi teks acak di ``docker logs``.
* Log yang bagus itu bukan untuk dibaca manusia satu-satu, tapi untuk **ditanya**.

Satu baris JSON = satu kejadian. Karena bentuknya JSON, log ini bisa langsung
dibaca pandas dan berubah jadi laporan drift::

    pd.read_json("logs/predictions.log", lines=True)

Itu yang dipakai ``src/monitor.py`` nanti.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.utils import config

# Field bawaan LogRecord. Apa pun di luar daftar ini berarti dikirim pemanggil
# lewat extra={...}, jadi ikut ditulis ke JSON.
_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JSONFormatter(logging.Formatter):
    """Mengubah satu catatan log jadi satu baris JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger(name: str = "house_price", to_file: bool = True) -> logging.Logger:
    """Logger yang menulis JSON ke layar dan (opsional) ke logs/predictions.log.

    Handler hanya dipasang sekali per nama logger — kalau tidak, memanggil
    fungsi ini dua kali membuat tiap baris log tercetak dobel.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.get_settings().log_level.upper(), logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(JSONFormatter())
    logger.addHandler(console)

    if to_file:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.PREDICTION_LOG, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    # propagate=False supaya log tidak ikut tercetak lagi oleh root logger.
    logger.propagate = False
    return logger
