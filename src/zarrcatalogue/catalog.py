# src/zarrcatalogue/catalog.py
from pathlib import Path
from typing import Dict, List, Optional, Union
import zarr
import json
from datetime import datetime
import pandas as pd
import numpy as np
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Union, Callable
import glob
from tqdm import tqdm

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class SimulationCatalog:
    """A catalog system for managing model simulation results."""
    
    def __init__(self, base_path: Path):
        """Initialize the catalog."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.simulations_path = self.base_path / "simulations"
        self.simulations_path.mkdir(exist_ok=True)
        self.index_file = self.base_path / "catalog.json"
        self._load_index()

    def _load_index(self):
        """Load or create the catalog index."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupted index file found. Creating new index.")
                self.index = self._create_new_index()
                self._save_index()
        else:
            self.index = self._create_new_index()
            self._save_index()

    def _create_new_index(self):
        """Create a new catalog index."""
        return {
            "simulations": {},
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }

    def _save_index(self):
        """Save the catalog index."""
        try:
            self.index["last_updated"] = datetime.now().isoformat()
            
            # Write to temporary file first
            temp_file = self.index_file.with_suffix('.json.tmp')
            with open(temp_file, 'w') as f:
                json.dump(self.index, f, indent=2, cls=CustomJSONEncoder)
            
            # Replace the original file
            temp_file.replace(self.index_file)
            
        except Exception as e:
            print(f"Error saving index: {e}")
            raise

    def add_simulation(
        self, 
        sim_id: str, 
        source_file: Path,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        converter: str = "mike"
    ) -> Dict:
        """Add a simulation to the catalog."""
        try:
            if sim_id in self.index["simulations"]:
                raise ValueError(f"Simulation ID {sim_id} already exists")

            # Create simulation directory
            sim_path = self.simulations_path / sim_id
            sim_path.mkdir(exist_ok=True)
            zarr_path = sim_path / "data.zarr"


            # Convert data using appropriate converter
            if converter.lower() == "mike":
                from .converters.mike import MIKEConverter
                converter = MIKEConverter()
            else:
                raise ValueError(f"Unknown converter: {converter}")

            # Convert to Zarr
            conversion_metadata = converter.to_zarr(source_file, zarr_path)

            # Create catalog entry
            entry = {
                "id": sim_id,
                "source_file": str(source_file),
                "zarr_store": str(zarr_path),
                "added_date": datetime.now().isoformat(),
                "converter": converter.model_type,
                "converter_version": converter.version,
                "conversion_metadata": conversion_metadata,
                "user_metadata": metadata or {},
                "tags": tags or [],
            }

            self.index["simulations"][sim_id] = entry
            self._save_index()
            return entry
            
        except Exception as e:
            # Clean up if something goes wrong
            if 'sim_path' in locals() and sim_path.exists():
                shutil.rmtree(sim_path)
            raise


    def _process_single_file(
        self,
        file_path: Path,
        metadata_generator: Optional[Callable] = None,
        tags: Optional[List[str]] = None,
        converter: str = "mike"
    ) -> Dict:
        """Process a single file for bulk import.
        
        Args:
            file_path: Path to the file
            metadata_generator: Optional function to generate metadata from file
            tags: List of tags to apply
            converter: Converter type to use
        """
        try:
            # Generate simulation ID from filename
            sim_id = file_path.stem
            
            # Generate metadata if function provided
            metadata = metadata_generator(file_path) if metadata_generator else {}
            
            # Add simulation to catalog
            entry = self.add_simulation(
                sim_id=sim_id,
                source_file=file_path,
                metadata=metadata,
                tags=tags,
                converter=converter
            )
            
            return {"success": True, "entry": entry, "file": str(file_path)}
            
        except Exception as e:
            return {
                "success": False,
                "file": str(file_path),
                "error": str(e)
            }

    def bulk_import(
        self,
        source_dir: Path,
        pattern: str = "*.dfsu",
        metadata_generator: Optional[Callable] = None,
        tags: Optional[List[str]] = None,
        converter: str = "mike",
        parallel: bool = True,
        max_workers: int = None,
        skip_existing: bool = True
    ) -> Dict[str, List]:
        """Import multiple simulations in parallel.
        
        Args:
            source_dir: Directory containing simulation files
            pattern: File pattern to match (e.g., "*.dfsu")
            metadata_generator: Optional function to generate metadata from files
            tags: List of tags to apply to all imported simulations
            converter: Converter type to use
            parallel: Whether to use parallel processing
            max_workers: Maximum number of parallel workers
            skip_existing: Skip files that would create duplicate simulation IDs
            
        Returns:
            Dictionary containing successful and failed imports
        """
        source_dir = Path(source_dir)
        files = list(source_dir.glob(pattern))
        
        if not files:
            raise ValueError(f"No files matching pattern '{pattern}' found in {source_dir}")

        print(f"Found {len(files)} files to process")
        
        # Check for potential duplicates if skip_existing is True
        if skip_existing:
            files = [f for f in files if f.stem not in self.index["simulations"]]
            print(f"{len(files)} files remaining after removing existing entries")

        results = {
            "successful": [],
            "failed": [],
            "skipped": []
        }

        if parallel and len(files) > 1:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all files for processing
                future_to_file = {
                    executor.submit(
                        self._process_single_file,
                        file_path,
                        metadata_generator,
                        tags,
                        converter
                    ): file_path 
                    for file_path in files
                }
                
                # Process results as they complete with progress bar
                with tqdm(total=len(files), desc="Processing files") as pbar:
                    for future in as_completed(future_to_file):
                        result = future.result()
                        if result["success"]:
                            results["successful"].append(result["entry"])
                        else:
                            results["failed"].append(result)
                        pbar.update(1)
        else:
            # Process files sequentially with progress bar
            for file_path in tqdm(files, desc="Processing files"):
                result = self._process_single_file(
                    file_path,
                    metadata_generator,
                    tags,
                    converter
                )
                if result["success"]:
                    results["successful"].append(result["entry"])
                else:
                    results["failed"].append(result)

        # Print summary
        print("\nImport Summary:")
        print(f"Successful imports: {len(results['successful'])}")
        print(f"Failed imports: {len(results['failed'])}")
        if results['failed']:
            print("\nFailed files:")
            for failure in results['failed']:
                print(f"- {failure['file']}: {failure['error']}")

        return results




    def get_simulation(self, sim_id: str) -> zarr.Group:
        """Get a simulation's Zarr store.
        
        Args:
            sim_id: Simulation identifier
            
        Returns:
            Zarr store containing the simulation data
        """
        if sim_id not in self.index["simulations"]:
            raise KeyError(f"Simulation {sim_id} not found")
            
        zarr_path = self.index["simulations"][sim_id]["zarr_store"]
        return zarr.open(zarr_path, 'r')

    def search(
        self,
        geometry_type: Optional[str] = None,
        variables: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        metadata_filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Search the catalog using various criteria.
        
        Args:
            geometry_type: Type of geometry to filter by
            variables: Required variables
            tags: Required tags
            metadata_filters: Filters to apply to metadata
            
        Returns:
            DataFrame of matching simulations
        """
        results = []
        
        for sim_id, entry in self.index["simulations"].items():
            # Check geometry type
            if geometry_type and entry["conversion_metadata"]["geometry_type"] != geometry_type:
                continue
                
            # Check variables
            sim_vars = entry["conversion_metadata"]["variables"]
            if variables and not all(var in sim_vars for var in variables):
                continue
                
            # Check tags
            if tags and not all(tag in entry["tags"] for tag in tags):
                continue
                
            # Check metadata filters
            if metadata_filters:
                match = True
                for key, value in metadata_filters.items():
                    if entry["user_metadata"].get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            results.append({
                "simulation_id": sim_id,
                **entry
            })
        
        return pd.DataFrame(results)

    def get_summary(self) -> Dict:
        """Get a summary of the catalog contents."""
        n_sims = len(self.index["simulations"])
        
        # Collect unique values
        geometry_types = set()
        variables = set()
        tags = set()
        
        for entry in self.index["simulations"].values():
            geometry_types.add(entry["conversion_metadata"]["geometry_type"])
            variables.update(entry["conversion_metadata"]["variables"])
            tags.update(entry["tags"])
        
        return {
            "n_simulations": n_sims,
            "geometry_types": sorted(geometry_types),
            "variables": sorted(variables),
            "tags": sorted(tags),
            "last_updated": self.index["last_updated"]
        }