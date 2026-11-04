"""
convert_sbx.py
Converts Scanbox (.sbx) two-photon imaging files into HDF5 format for Suite2p ingestion.
This version automatically extracts and interleaves 2 channels (Green and Red) from a 
single focal plane so Suite2p can use the second channel for Cellpose and registration.

Usage:
    python pipelines\preprocessing\convert_sbx.py --input data\mouse_6416_1R1L\day_1\raw\6416_1R1L_000_000.sbx --output data\mouse_6416_1R1L\day_1\raw\6416_1R1L_000_000.h5
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
try:
    import sbxreader
except ImportError:
    raise ImportError("The 'sbxreader' package is required. Run: pip install sbxreader")

def convert_sbx_to_suite2p_h5(sbx_path: Path, h5_path: Path):
    sbx_path = Path(sbx_path)
    
    if not sbx_path.exists():
        raise FileNotFoundError(f"Could not find Scanbox file: {sbx_path}")

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading Scanbox memory map for: {sbx_path.name}...")
    
    sbx_mmap = sbxreader.sbx_memmap(str(sbx_path))
    shape = sbx_mmap.shape
    print(f"Detected array shape: {shape}")

    # Assumes shape is (Frames, Planes, Channels, Y, X) with 1 Plane and 2 Channels
    if len(shape) == 5:
        n_frames, n_planes, n_channels, Ly, Lx = shape
    else:
        raise ValueError(f"Unexpected sbx shape for 2-channel data: {shape}")

    if n_channels != 2:
        raise ValueError(f"Expected 2 channels, but found {n_channels}.")

    print(f"Interleaving 2 channels into {h5_path.name}...")
    chunk_size = 1000 
    
    with h5py.File(h5_path, 'w') as h5f:
        # Double the frame count to hold both channels interleaved
        dset = h5f.create_dataset(
            "data", 
            shape=(n_frames * 2, Ly, Lx), 
            dtype=sbx_mmap.dtype, 
            chunks=(min(n_frames * 2, 100), Ly, Lx)
        )
        
        for i in range(0, n_frames, chunk_size):
            end_idx = min(i + chunk_size, n_frames)
            
            # Extract Green (Chan 0) and Red (Chan 1)
            # Using plane index 0
            ch0_chunk = sbx_mmap[i:end_idx, 0, 0, :, :]
            ch1_chunk = sbx_mmap[i:end_idx, 0, 1, :, :]
            
            # Interleave the arrays
            chunk_frames = end_idx - i
            interleaved = np.empty((chunk_frames * 2, Ly, Lx), dtype=sbx_mmap.dtype)
            interleaved[0::2, :, :] = ch0_chunk
            interleaved[1::2, :, :] = ch1_chunk
            
            dset[i*2:end_idx*2, :, :] = interleaved
            print(f"  -> Processed frames {i} to {end_idx} / {n_frames}")

    print(f"\n✓ Conversion complete! Interleaved HDF5 saved to: {h5_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Scanbox (.sbx) to Suite2p HDF5 (.h5)")
    parser.add_argument("--input", type=str, required=True, help="Path to input Scanbox .sbx file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .h5 file")
    args = parser.parse_args()

    convert_sbx_to_suite2p_h5(Path(args.input), Path(args.output))

if __name__ == "__main__":
    main()