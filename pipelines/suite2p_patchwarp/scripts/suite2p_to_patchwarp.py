"""
suite2p_to_patchwarp.py
Extracts Suite2p mean templates and ROI spatial footprints from multiple
session folders and exports them to MATLAB .mat files for PatchWarp across-session registration.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio


def process_suite2p_session(session_dir: Path, out_dir: Path, session_name: str, only_cells: bool = True):
    """
    Parses a single suite2p plane0 directory and saves a clean .mat file for PatchWarp.
    """
    plane0_path = session_dir / "suite2p" / "plane0"
    if not plane0_path.exists():
        # Fallback in case path given was already plane0
        plane0_path = session_dir if (session_dir / "ops.npy").exists() else None

    if plane0_path is None or not (plane0_path / "ops.npy").exists():
        raise FileNotFoundError(f"Could not find valid Suite2p outputs (ops.npy) in {session_dir}")

    print(f"\nProcessing {session_name} from: {plane0_path}")

    # 1. Load Suite2p outputs
    ops = np.load(plane0_path / "ops.npy", allow_pickle=True).item()
    stat = np.load(plane0_path / "stat.npy", allow_pickle=True)
    iscell = np.load(plane0_path / "iscell.npy", allow_pickle=True)

    Ly = int(ops["Ly"])
    Lx = int(ops["Lx"])
    mean_img = ops["meanImg"].astype(np.float32)
    mean_img_e = ops.get("meanImgE", mean_img).astype(np.float32)

    # 2. Filter ROIs (keep only curated cells if specified)
    if only_cells:
        cell_indices = np.where(iscell[:, 0] == 1)[0]
    else:
        cell_indices = np.arange(len(stat))

    n_rois = len(cell_indices)
    print(f"Found {len(stat)} total ROIs. Selected {n_rois} ROIs for alignment.")

    # 3. Construct 2D coordinate lists & 3D binary footprint tensor (Ly, Lx, N)
    # PatchWarp can ingest spatial footprints directly as masks or 2D centroids
    roi_masks = np.zeros((Ly, Lx, n_rois), dtype=np.uint8)
    centroids = np.zeros((n_rois, 2), dtype=np.float32)  # [y, x]
    roi_pixel_lists = []

    for i, idx in enumerate(cell_indices):
        ypix = stat[idx]["ypix"]
        xpix = stat[idx]["xpix"]
        roi_masks[ypix, xpix, i] = 1
        centroids[i, 0] = stat[idx]["med"][0]  # y-center
        centroids[i, 1] = stat[idx]["med"][1]  # x-center
        # 1-indexed for MATLAB coordinates
        roi_pixel_lists.append(np.column_stack([xpix + 1, ypix + 1]))

    # 4. Save to MATLAB-compatible dictionary
    mat_payload = {
        "session_name": session_name,
        "Ly": Ly,
        "Lx": Lx,
        "meanImg": mean_img,
        "meanImgE": mean_img_e,
        "centroids": centroids,
        "roi_indices": cell_indices + 1,  # 1-indexed for MATLAB
        "roi_masks": roi_masks
    }

    out_file = out_dir / f"{session_name}_patchwarp_prep.mat"
    sio.savemat(out_file, mat_payload, do_compression=True)
    print(f"Saved: {out_file.name} ({out_file.stat().st_size / (1024 * 1024):.2f} MB)")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Export Suite2p outputs for PatchWarp across-session registration.")
    parser.add_argument(
        "--sessions",
        nargs="+",
        required=True,
        help="Paths to your 3 day session directories (e.g. C:/data/day1 C:/data/day2 C:/data/day3)"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=["day1", "day2", "day3"],
        help="Identifier names for each session"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./patchwarp_inputs",
        help="Target folder for export files"
    )
    parser.add_argument(
        "--all_rois",
        action="store_true",
        help="Include all detected ROIs instead of only iscell==1"
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(args.sessions) != len(args.names):
        raise ValueError("Number of session paths must match number of session names.")

    for s_path, s_name in zip(args.sessions, args.names):
        process_suite2p_session(Path(s_path), out_dir, s_name, only_cells=not args.all_rois)

    print("\n✓ All session summary files successfully exported for PatchWarp.")


if __name__ == "__main__":
    main()
