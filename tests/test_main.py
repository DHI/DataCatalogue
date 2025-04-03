import zarrcatalogue

def test_package_has_version() -> None:
    assert zarrcatalogue.__version__ is not None

    major, minor, _ = zarrcatalogue.__version__.split(".")

    assert isinstance(int(major),int)
    assert isinstance(int(minor),int)