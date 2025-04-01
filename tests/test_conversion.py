import mikeio
import numpy as np
import pandas as pd
import zarr
import zarr.storage
from zarrcatalogue.converters.mike import MIKEConverter


def test_to_from_zarr(tmp_path) -> None:
    fp = "tests/testdata/oresundHD_run1.dfsu"
    out_path = tmp_path / "oresund.zarr"
    dfsu_path = tmp_path / "oresund.dfsu"

    converter = MIKEConverter()
    converter.to_zarr(fp, out_path)

    converter.from_zarr(out_path, dfsu_path)

    ds1 = mikeio.read(fp)
    ds2 = mikeio.read(dfsu_path)

    assert np.allclose(ds1.geometry.node_coordinates, ds2.geometry.node_coordinates)
    assert np.allclose(ds1["Surface elevation"].values, ds2["Surface elevation"].values)
    assert ds1["Surface elevation"].type == ds2["Surface elevation"].type
    assert ds1["Surface elevation"].unit == ds2["Surface elevation"].unit
    for da1, da2 in zip(ds1, ds2):
        assert np.allclose(da1.values, da2.values)


def test_to_from_long_timeseries(tmp_path) -> None:
    time = pd.date_range("2022-01-01", "2023-12-31", freq="30min")

    geometry = mikeio.Grid2D(nx=2, ny=2, dx=1, dy=1).to_geometryFM()

    ds = mikeio.DataArray(
        data=np.zeros((len(time), geometry.n_elements)), time=time, geometry=geometry
    )._to_dataset()

    zarr_fp = zarr.storage.MemoryStore()
    dfsc_fp = tmp_path / "converted.dfsu"

    converter = MIKEConverter()
    converter.to_zarr(ds, zarr_fp)
    converter.from_zarr(zarr_fp, dfsc_fp)
