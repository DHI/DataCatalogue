import mikeio
import numpy as np
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

    # reorder based on names
    ds3 = ds2['Surface elevation', 'Total water depth', 'U velocity', 'V velocity']    
    for da1, da2 in zip(ds1, ds3):
        assert np.allclose(da1.values, da2.values)

