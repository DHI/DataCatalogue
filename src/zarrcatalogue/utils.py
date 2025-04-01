import os
import numpy as np


# Check number of files and total size of Zarr output in zarr_path
def analyze_zarr_storage(zarr_path):
    """
    Analyze a Zarr directory to count files and calculate total size.
    
    Args:
        zarr_path: Path to the Zarr store
        
    Returns:
        Dictionary with statistics about the Zarr store
    """
    # Initialize counters
    total_size = 0
    file_count = 0
    file_extensions = {}
    largest_files = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(zarr_path):
        for file in files:
            # Skip .zarray and .zattrs metadata files if you want just data files
            # if file in ['.zarray', '.zattrs']:
            #     continue
                
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            
            # Count files
            file_count += 1
            
            # Add to total size
            total_size += file_size
            
            # Track file extensions
            ext = os.path.splitext(file)[1]
            if ext in file_extensions:
                file_extensions[ext] += 1
            else:
                file_extensions[ext] = 1
                
            # Track largest files
            largest_files.append((file_path, file_size))
            largest_files = sorted(largest_files, key=lambda x: x[1], reverse=True)[:5]
    
    # Convert bytes to more readable format
    def format_size(size_bytes):
        """Convert size in bytes to human-readable format"""
        if size_bytes == 0:
            return "0 B"
        size_names = ("B", "KB", "MB", "GB", "TB")
        i = int(np.floor(np.log(size_bytes) / np.log(1024)))
        p = np.power(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    # Prepare results
    results = {
        "total_files": file_count,
        "total_size_bytes": total_size,
        "total_size_formatted": format_size(total_size),
        "file_extensions": file_extensions,
    }
    
    return results