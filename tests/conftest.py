"""Fixture bersama untuk seluruh test."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.data_loader import load_train_holdout
from src.models.predict import CONTOH_REQUEST


@pytest.fixture(scope="session")
def data_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/holdout. scope=session supaya data cuma dibaca sekali."""
    return load_train_holdout()


@pytest.fixture
def request_valid() -> dict:
    return dict(CONTOH_REQUEST)
