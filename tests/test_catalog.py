from pathlib import Path

from zarrcatalogue.catalog import SimulationCatalog


def test_search_metadata(tmp_path: Path) -> None:

    catalog = SimulationCatalog(tmp_path)
    catalog.add_simulation("oresundHD_run1", source_file=Path("tests/testdata/oresundHD_run1.dfsu"), metadata={"engine": "MIKE21"})
    filtered = catalog.search(metadata_filters={"engine": "MIKE21"})
    assert len(filtered) == 1

    no_luck = catalog.search(metadata_filters={"engine": "OpenFoam"})
    assert len(no_luck) == 0

def test_summary(tmp_path: Path) -> None:

    catalog = SimulationCatalog(tmp_path)
    catalog.add_simulation("oresundHD_run1", source_file=Path("tests/testdata/oresundHD_run1.dfsu"))
    summary = catalog.get_summary()
    assert summary["n_simulations"] == 1