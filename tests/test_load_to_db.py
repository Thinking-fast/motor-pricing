import pandas as pd

from src.data.load_to_db import _build_regions


def test_build_regions_deduplicates_and_names_known_codes():
    freq = pd.DataFrame({"Region": ["R82", "R11", "R82"]})

    regions = _build_regions(freq)

    assert len(regions) == 2
    assert set(regions.columns) == {"region", "region_name", "macro_area"}

    names = dict(zip(regions["region"], regions["region_name"]))
    assert names["R11"] == "Ile-de-France"


def test_build_regions_falls_back_to_the_raw_code():
    freq = pd.DataFrame({"Region": ["R99"]})

    regions = _build_regions(freq)

    assert regions["region_name"].iloc[0] == "R99"
