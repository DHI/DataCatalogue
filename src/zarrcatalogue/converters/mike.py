from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import mikeio
import zarr
import numpy as np
from pydantic import BaseModel

MODEL_TYPE = "MIKE"

# TODO move version
version = "0.1.0"
class ItemAttributes(BaseModel):
    unit: mikeio.EUMUnit
    item_info: mikeio.EUMType
    item_number: int

def _process_element_table(element_table: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Process element table and return formatted data and metadata.
    
    Args:
        element_table: Array of element connectivity
        
    Returns:
        Tuple containing:
        - Formatted element table (padded with -1)
        - Dictionary of element metadata
    """
    # Analyze element types
    element_lengths = [len(elem) for elem in element_table]
    unique_lengths = np.unique(element_lengths)
    
    # Count elements of each type
    element_counts = {
        f'n_elements_{length}_nodes': sum(1 for x in element_lengths if x == length)
        for length in unique_lengths
    }
    
    # Create padded array
    max_nodes = max(unique_lengths)
    formatted_table = np.full((len(element_table), max_nodes), -1, dtype=np.int32)
    
    # Fill the array
    for i, elem in enumerate(element_table):
        formatted_table[i, :len(elem)] = elem
        
    # Create metadata
    element_metadata = {
        'max_nodes_per_element': int(max_nodes),
        'min_nodes_per_element': int(min(unique_lengths)),
        'element_types_present': sorted(unique_lengths),
        **element_counts
    }
    
    return formatted_table, element_metadata

def to_zarr(
    input_file: Path, 
    zarr_path: Path, 
    chunks: Optional[Dict] = None,
    compression_level: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """Convert MIKE dfsu file to zarr format.
    
    Args:
        input_file: Path to dfsu file
        zarr_path: Output zarr store path
        chunks: Dictionary of chunk sizes {'time': int, 'elements': int}
        compression_level: Level of compression (1-9)
        **kwargs: Additional conversion options
    
    Returns:
        Dictionary containing metadata about the conversion
    """
    # Read MIKE file
    ds = mikeio.read(input_file)
    
    # Default chunking if not specified
    if chunks is None:
        chunks = {'time': min(100, ds.n_timesteps), 'elements': -1}
    
    # Create zarr store
    store: zarr.Group = zarr.open(zarr_path, mode='w')
    
    # Store mesh topology
    topo: zarr.Group = store.create_group('topology')
    
    # Store geometry information
    #topo.create_array('nodes', data=ds.geometry.node_coordinates)
    topo['nodes'] = ds.geometry.node_coordinates
    
    # Process and store element table
    formatted_elements, element_metadata = _process_element_table(ds.geometry.element_table)
    topo['elements'] = formatted_elements
    
    # Store element coordinates
    topo['element_coordinates'] = ds.geometry.element_coordinates
    
    # Store geometry metadata
    geometry_metadata = {
        'projection': str(ds.geometry.projection),
        'is_2d': getattr(ds.geometry, 'is_2d', True),
        'n_elements': ds.geometry.n_elements,
        'n_nodes': ds.geometry.n_nodes,
        'geometry_type': ds.geometry.__class__.__name__,
    }
    geometry_metadata.update(element_metadata)
    
    # Add 3D specific information
    if isinstance(ds.geometry, mikeio.spatial._FM_geometry_layered.GeometryFMVerticalProfile):
        geometry_metadata.update({
            'vertical_profile': True,
            'n_sigma_layers': getattr(ds.geometry, 'n_sigma', None),
            'n_layers': getattr(ds.geometry, 'n_layers', None)
        })
    
    topo.attrs.update(geometry_metadata)
    
    # Create data group
    data = store.create_group('data')
    
    # Store time information
    time_stamps = np.array([t.timestamp() for t in ds.time])
    data['time'] = time_stamps
    
    # Add time metadata
    data.attrs.update({
        'start_time': str(ds.start_time),
        'end_time': str(ds.end_time),
        'timestep': str(ds.timestep) if ds.is_equidistant else 'non-equidistant',
        'n_timesteps': ds.n_timesteps
    })
    
    # Store each item's data
    for item_number, da in enumerate(ds):
        item_data = da.to_numpy()
        
        # Determine chunks based on data shape
        # item_chunks = (chunks['time'], chunks['elements'])
        #if item_data.ndim > 2:  # For vector quantities
        #    item_chunks = item_chunks + (-1,)
        
        # Create dataset with compression
        data.create_array(
            da.name,
            shape=item_data.shape,
            dtype=item_data.dtype,
            #data=item_data,
        #    chunks=item_chunks,
            #compression='blosc',
            #compression_opts={'cname': 'zstd', 'clevel': compression_level}
        )
        data[da.name] = item_data
        
        # Store item metadata
        data[da.name].attrs.update({
            'unit': da.unit,
            'item_info': str(da.type),
            'item_number': item_number
        })
    
    # Store conversion metadata
    conversion_metadata = {
        "model_type": MODEL_TYPE,
        "converter_version": version,
        "conversion_time": datetime.now().isoformat(),
        "input_file": str(input_file),
        "mikeio_version": mikeio.__version__,
        "geometry_type": ds.geometry.__class__.__name__,
        "n_elements": ds.geometry.n_elements,
        "n_nodes": ds.geometry.n_nodes,
        "n_timesteps": ds.n_timesteps,
        "variables": ds.names,
        "time_range": [str(ds.start_time), str(ds.end_time)],
        "element_info": element_metadata,
        "chunks": chunks,
        "compression_level": compression_level
    }
    
    # Store conversion metadata in root group
    store.attrs.update(conversion_metadata)
    
    return conversion_metadata

def from_zarr(
    zarr_path: Path,
    output_file: Path,
) -> Dict[str, Any]:
    """Convert zarr store back to MIKE dfsu format.
    
    Args:
        zarr_path: Path to input zarr store
        output_file: Path to output dfsu file
        
    Returns:
        Dictionary containing metadata about the conversion
    """
    # Open zarr store
    store = zarr.open(zarr_path, mode='r')
    
    # Extract geometry information
    topo = store['topology']
    nodes = topo['nodes'][:]
    elements: np.ndarray = topo['elements'][:]
    
    # Remove padding (-1) from elements
    elements = np.array([elem[elem >= 0] for elem in elements])
    
    # Create appropriate geometry based on metadata
    geometry_type = store.attrs['geometry_type']
    if geometry_type == 'GeometryFM2D':
        geometry = mikeio.spatial.GeometryFM2D(
            node_coordinates=nodes,
            element_table=elements,
            projection=topo.attrs.get('projection')
        )
    elif geometry_type == 'GeometryFMVerticalProfile':
        geometry = mikeio.spatial.GeometryFMVerticalProfile(
            node_coordinates=nodes,
            element_table=elements,
            n_sigma=topo.attrs.get('n_sigma_layers'),
            n_layers=topo.attrs.get('n_layers'),
            projection=topo.attrs.get('projection')
        )
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")
    
    # Extract time information
    time_data: np.ndarray = store['data/time'][:]
    time = [datetime.fromtimestamp(t) for t in time_data]
    
    # Create data arrays for each variable
    items = []
    item_map_order = {}
    data_members = [k for k in store['data'].array_keys() if k != 'time']
    for item_name in data_members:
        item_data = store['data'][item_name][:]
        info = ItemAttributes(**store['data'][item_name].attrs)
        item_map_order[item_name] = info.item_number
        
        items.append(mikeio.DataArray(
            data=item_data,
            time=time,
            geometry=geometry,
            item=mikeio.ItemInfo(item_name, info.item_info, info.unit)
        ))
    
    # Create dataset
    ds = mikeio.Dataset(items)

    item_names = sorted(item_map_order, key=lambda k: item_map_order[k])
    ds_sorted = ds[item_names]
    ds_sorted.to_dfs(output_file)
    
    # Return metadata about the conversion
    conversion_metadata = {
        "model_type": MODEL_TYPE,
        "converter_version": version,
        "conversion_time": datetime.now().isoformat(),
        "input_file": str(zarr_path),
        "output_file": str(output_file),
        "mikeio_version": mikeio.__version__,
        "geometry_type": geometry_type,
        "n_elements": len(elements),
        "n_nodes": len(nodes),
        "n_timesteps": len(time),
        "variables": [item.name for item in items],
        "time_range": [str(time[0]), str(time[-1])]
    }
    
    return conversion_metadata

def validate_conversion(
    original_ds: Union[mikeio.Dataset, Path], 
    zarr_path: Path
) -> Dict[str, bool]:
    """Validate the conversion results.
    
    Args:
        original_ds: Original MIKE dataset or path to dfsu file
        zarr_path: Path to converted Zarr store
        
    Returns:
        Dictionary of validation results
    """
    if isinstance(original_ds, (str, Path)):
        original_ds = mikeio.read(original_ds)
        
    store = zarr.open(zarr_path, 'r')
    
    validations = {
        'element_count_match': original_ds.geometry.n_elements == store['topology'].attrs['n_elements'],
        'node_count_match': original_ds.geometry.n_nodes == store['topology'].attrs['n_nodes'],
        'time_steps_match': len(original_ds.time) == len(store['data/time']),
        'variables_match': set(original_ds.names) == set(name for name in store['data'].array_keys() if name != 'time'),
        'geometry_type_match': original_ds.geometry.__class__.__name__ == store.attrs['geometry_type']
    }
    
    # Add detailed validation results
    validations['all_valid'] = all(validations.values())
    
    return validations

def get_model_info() -> Dict[str, Any]:
    """Return information about the MIKE converter."""
    return {
        "model_type": MODEL_TYPE,
        "converter_version": version,
        "supported_formats": ["dfsu"],
        "mikeio_version": mikeio.__version__
    }