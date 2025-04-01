# src/zarrcatalogue/plotting.py
from pathlib import Path
import mikeio
from mikeio.spatial import GeometryFMVerticalProfile
import zarr
from datetime import datetime

class MeshPlotter:
    """Plotting utilities using mikeio's native plotting capabilities."""
    
    def __init__(self, zarr_path: Path):
        """Initialize plotter with Zarr store path."""
        self.zarr_path = zarr_path
        self.store = zarr.open(zarr_path)
        
    def to_mikeio_dataset(self) -> mikeio.Dataset:
        """Convert Zarr store back to mikeio Dataset."""
        # Get all variables except time
        variables = [k for k in self.store['data'].array_keys() if k != 'time']
        
        # Convert timestamps to datetime
        timestamps = [datetime.fromtimestamp(t) for t in self.store['data/time'][:]]
        
        # Create geometry
        nodes = self.store['topology/nodes'][:]
        elements = self.store['topology/elements'][:]
        
        # Create geometry object with all required attributes
        geometry = GeometryFMVerticalProfile(
            node_coordinates=nodes,
            element_table=elements,
            n_sigma_layers=self.store['topology'].attrs.get('n_sigma_layers', 10),
            projection=self.store['topology'].attrs.get('projection', 'UTM-32')
        )
        
        # Create DataArray for each variable
        data_items = []
        for var in variables:
            data = self.store[f'data/{var}'][:]
            data_items.append(
                mikeio.DataArray(
                    data=data,
                    time=timestamps,
                    geometry=geometry,
                    item=var
                )
            )
        
        # Create Dataset
        ds = mikeio.Dataset(data_items)
        return ds
    
    def plot(self, variable: str, n: int = 0, **kwargs):
        """Plot using mikeio's native plotting.
        
        Args:
            variable: Name of the variable to plot
            n: Time step index to plot
            **kwargs: Additional arguments passed to mikeio's plot function
        """
        ds = self.to_mikeio_dataset()
        # Use mikeio's native plotting
        ds[variable].plot(n=n, **kwargs)
