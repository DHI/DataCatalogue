# src/zarrcatalogue/converters/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

class BaseConverter(ABC):
    """Base class for all converters."""
    
    @abstractmethod
    def to_zarr(self, input_file: Path, zarr_path: Path, **kwargs) -> Dict[str, Any]:
        """Convert input file to zarr format."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the converter."""
        pass

    @abstractmethod
    def validate_conversion(self, original_data: Any, zarr_path: Path) -> Dict[str, bool]:
        """Validate the conversion results."""
        pass