# zarrcatalogue

A Python package for converting and managing model results (e.g. flexible mesh from MIKE) using Zarr storage format.


## repository structure

zarrcatalogue/
├── src/
│   ├── zarrcatalogue/
│   │   ├── __init__.py
│   │   ├── converters/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── mike.py
│   │   ├── catalog.py
│   │   └── manager.py
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_mike_converter.py
│   └── setup.py
├── notebooks/
│   ├── 01_basic_conversion.ipynb
│   ├── 02_metadata_handling.ipynb
│   └── 03_querying_catalog.ipynb
└── README.md

data/
data_zarr/ # only temporary for testing of conversion
catalog/ # 


## Features

### MIKE Model Converter
Currently supports conversion of MIKE flexible mesh files (dfsu) to Zarr format, handling:
- 2D and 3D flexible mesh geometries
- Mixed element types (triangular and quadrilateral elements)
- Multiple variables and time steps
- Mesh topology and element data
- Comprehensive metadata storage

#### Data Structure
The converted Zarr store follows this structure:

simulation.zarr/
├── data/
│ ├── variable1 # (n_timesteps, n_elements) array
│ ├── variable2 # (n_timesteps, n_elements) array
│ └── time # (n_timesteps,) array of timestamps
└── topology/
├── nodes # (n_nodes, 3) node coordinates
├── elements # (n_elements, max_nodes) connectivity
└── element_coordinates # (n_elements, 3) element centers


#### Metadata
Stores comprehensive metadata including:
- Model information (type, version)
- Geometry details (nodes, elements, projection)
- Mesh characteristics (element types, counts)
- Time information (start, end, timestep)
- Variable attributes (units, descriptions)
- Conversion details (timestamp, software versions)

#### Performance Features
- Configurable chunking for efficient data access
- Compression options for reduced storage
- Optimized for both temporal and spatial queries

## Usage

Basic conversion:
```python
from zarrcatalogue.converters.mike import MIKEConverter
from pathlib import Path

# Initialize converter
converter = MIKEConverter()

# Convert MIKE dfsu file to Zarr
metadata = converter.to_zarr(
    input_file=Path("simulation.dfsu"),
    zarr_path=Path("output.zarr"),
    chunks={'time': 100, 'elements': 1000},
    compression_level=5
)

# Validate conversion
validation = converter.validate_conversion(
    original_ds="simulation.dfsu",
    zarr_path=Path("output.zarr")
)
```

## Requirements
* Python 3.x
* mikeio >= 2.2.0
* zarr
* numpy

## Current Limitations
Currently supports MIKE dfsu files only
Element types limited to triangles and quadrilaterals
Single file conversion (no batch processing yet)

## Future Development

### Planned features:
* conversion back zarr2mike
* leverage mikeio plotting after conversion back
* Batch processing capabilities
* Data catalog system for managing multiple simulations
* Advanced querying and filtering

* Export capabilities to other formats
* Support for additional MIKE file formats (MIKE SHE, FEFLOW)
