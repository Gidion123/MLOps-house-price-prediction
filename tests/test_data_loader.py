"""Test pemuatan & validasi data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import data_loader
from src.utils import config


def test_pembacaan_jujur_menyelamatkan_kategori_none(tmp_path):
    """Kategori sah "None" tidak boleh ditelan jadi NaN oleh pandas."""
    berkas = tmp_path / "mini.csv"
    berkas.write_text("Id,MasVnrType\n1,None\n2,BrkFace\n3,NA\n", encoding="utf-8")

    default = pd.read_csv(berkas)
    jujur = data_loader.read_csv_honest(berkas)

    assert default["MasVnrType"].isna().sum() == 2      # pandas menelan "None"
    assert jujur["MasVnrType"].isna().sum() == 1        # hanya "NA" yang missing
    assert "None" in set(jujur["MasVnrType"].dropna())


def test_kolom_angka_tetap_numerik_setelah_dibaca_jujur(tmp_path):
    berkas = tmp_path / "mini.csv"
    berkas.write_text("Id,LotFrontage\n1,65\n2,NA\n3,80\n", encoding="utf-8")
    hasil = data_loader.read_csv_honest(berkas)
    assert pd.api.types.is_numeric_dtype(hasil["LotFrontage"])
    assert hasil["LotFrontage"].isna().sum() == 1


def test_validasi_menolak_tabel_kosong():
    with pytest.raises(data_loader.DataValidationError):
        data_loader.validate_raw(pd.DataFrame())


def test_validasi_menolak_target_tidak_masuk_akal(data_split):
    train, _ = data_split
    rusak = train.copy()
    rusak.loc[rusak.index[0], config.TARGET] = -1
    with pytest.raises(data_loader.DataValidationError, match="SalePrice"):
        data_loader.validate_raw(rusak)


def test_validasi_menolak_nilai_di_luar_rentang(data_split):
    train, _ = data_split
    rusak = train.copy()
    rusak.loc[rusak.index[0], "OverallQual"] = 99
    with pytest.raises(data_loader.DataValidationError, match="OverallQual"):
        data_loader.validate_raw(rusak)


def test_split_reproducible(data_split):
    train, holdout = data_split
    train2, holdout2 = data_loader.load_train_holdout()
    assert train[config.ID_COLUMN].tolist() == train2[config.ID_COLUMN].tolist()
    assert holdout[config.ID_COLUMN].tolist() == holdout2[config.ID_COLUMN].tolist()


def test_holdout_tidak_beririsan_dengan_train(data_split):
    train, holdout = data_split
    assert set(train[config.ID_COLUMN]).isdisjoint(set(holdout[config.ID_COLUMN]))


def test_holdout_bebas_kembar_fitur(data_split):
    """Ini yang menjaga skor holdout tetap jujur."""
    train, holdout = data_split
    sidik_train = set(data_loader.contract_signature(train))
    assert not data_loader.contract_signature(holdout).isin(sidik_train).any()
