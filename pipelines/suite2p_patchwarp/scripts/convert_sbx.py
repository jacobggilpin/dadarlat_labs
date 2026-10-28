"""
convert_sbx.py
Converts Scanbox (.sbx) two-photon imaging files into HDF5 format for Suite2p ingestion.

Usage:
    python pipelines/suite2p_patchwarp/scripts/convert_sbx.py --input data/mouse_6416_1R1L/day_1/6416_1R1L_000_000.sbx --output data/mouse_6416_1R1L/day_1/6416_1R1L_000_000.h5 --fs 15.625
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
try:
    import sbxreader
except ImportError:
    raise ImportError("The 'sbxreader' package is required. Run: pip install sbxreader")


def convert_sbx_to_suite2p_h5(sbx_path: Path, h5_path: Path, channel_idx: int = 0, plane_idx: int = 0):
    sbx_path = Path(sbx_path)
    mat_path = sbx_path.with_suffix('.mat')

    if not sbx_path.exists():
        raise FileNotFoundError(f"Could not find Scanbox file: {sbx_path}")
    if not mat_path.exists():
        raise FileNotFoundError(f"Could not find corresponding Scanbox metadata file: {mat_path}")

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading Scanbox memory map for: {sbx_path.name}...")
    
    sbx_mmap = sbxreader.sbx_memmap(str(sbx_path))
    shape = sbx_mmap.shape
    print(f"Detected array shape: {shape}")

    # Handle varying output dimensions from sbxreader (3D, 4D, or 5D)
    if len(shape) == 5:
        n_frames, n_planes, n_channels, Ly, Lx = shape
    elif len(shape) == 4:
        n_frames, n_channels, Ly, Lx = shape
        n_planes = 1
        plane_idx = 0
    elif len(shape) == 3:
        n_frames, Ly, Lx = shape
        n_channels = 1
        channel_idx = 0
        n_planes = 1
        plane_idx = 0
    else:
        raise ValueError(f"Unexpected sbx shape: {shape}")

    if channel_idx >= n_channels:
        raise ValueError(f"Requested channel {channel_idx} but data only has {n_channels} channels.")
    if plane_idx >= n_planes:
        raise ValueError(f"Requested plane {plane_idx} but data only has {n_planes} planes.")

    print(f"Extracting Plane {plane_idx}, Channel {channel_idx} and chunking to {h5_path.name}...")
    chunk_size = 1000 
    
    with h5py.File(h5_path, 'w') as h5f:
        dset = h5f.create_dataset(
            "data", 
            shape=(n_frames, Ly, Lx), 
            dtype=sbx_mmap.dtype, 
            chunks=(min(n_frames, 100), Ly, Lx)
        )
        
        for i in range(0, n_frames, chunk_size):
            end_idx = min(i + chunk_size, n_frames)
            
            # Slice the array dynamically based on its dimensions
            if len(shape) == 5:
                chunk_data = sbx_mmap[i:end_idx, plane_idx, channel_idx, :, :]
            elif len(shape) == 4:
                chunk_data = sbx_mmap[i:end_idx, channel_idx, :, :]
            else:
                chunk_data = sbx_mmap[i:end_idx, :, :]
            
            dset[i:end_idx, :, :] = chunk_data
            print(f"  -> Processed frames {i} to {end_idx} / {n_frames}")

    print(f"\n✓ Conversion complete! HDF5 saved to: {h5_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Scanbox (.sbx) to Suite2p HDF5 (.h5)")
    parser.add_argument("--input", type=str, required=True, help="Path to input Scanbox .sbx file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .h5 file")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to extract (default: 0)")
    parser.add_argument("--plane", type=int, default=0, help="Plane index to extract (default: 0)")
    parser.add_argument("--fs", type=float, default=15.625, help="Scanbox frame rate in Hz (default: 15.625)")
    args = parser.parse_args()

    convert_sbx_to_suite2p_h5(
        Path(args.input), 
        Path(args.output), 
        channel_idx=args.channel,
        plane_idx=args.plane
    )


if __name__ == "__main__":
    main()