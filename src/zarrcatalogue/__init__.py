from importlib.metadata import PackageNotFoundError, version

try:
    # read version from installed package
    # TODO is it zarrcatalogue or datacatalogue?
    __version__ = version("datacatalogue")
except PackageNotFoundError:
    # package is not installed 
    __version__ = "dev"
