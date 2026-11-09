"""
read_settings.py
Utility script to load and inspect Suite2p settings.npy or ops.npy files.

Example usage:
    python pipelines\core\read_settings.py data\mouse_6416_1R1L\day_1\run_02_sparsery\suite2p\settings.npy --keys threshold_scaling tau snr_thresh
"""

import argparse
import numpy as np
from pathlib import Path
import pprint

def inspect_settings(file_path: Path, keys: list = None):
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return

    try:
        ops = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path.name}: {e}")
        return

    print(f"\n=== Suite2p Settings Inspector ===")
    print(f"File: {file_path.resolve()}\n")

    if keys:
        for key in keys:
            val = ops.get(key, "KEY NOT FOUND")
            print(f"{key}: {val}")
    else:
        pprint.pprint(ops, width=80, sort_dicts=False)
        print("\nTip: To search specific variables, use the --keys argument.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Suite2p settings.npy or ops.npy files.")
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the settings.npy or ops.npy file"
    )
    parser.add_argument(
        "--keys", 
        nargs="+", 
        help="Specific settings to look up (e.g., --keys tau threshold_scaling snr_thresh)"
    )

    args = parser.parse_args()
    inspect_settings(Path(args.file_path), args.keys)