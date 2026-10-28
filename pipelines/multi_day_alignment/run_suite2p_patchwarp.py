"""
run_pipeline.py
One-click streamlined runner for:
  Stage 1: Suite2p template & mask extraction (Python)
  Stage 2: PatchWarp across-session affine registration (MATLAB R2022b)
  Stage 3: Longitudinal ROI matching & trace compilation (Python)

Example usage:
    python pipelines/suite2p_patchwarp/run_pipeline.py --mouse mouse_6416_1R1L --days day_1 day_2 day_3 --target day_2
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "scripts"))
from suite2p_to_patchwarp import process_suite2p_session
from match_longitudinal_rois import match_across_sessions


def main():
    parser = argparse.ArgumentParser(description="Streamlined Suite2p-to-PatchWarp Pipeline")
    parser.add_argument("--mouse", type=str, required=True, help="Mouse folder name inside data/ (e.g. mouse_6416_1R1L)")
    parser.add_argument("--days", nargs="+", default=["day_1", "day_2", "day_3"], help="Subfolder names for each day")
    parser.add_argument("--target", type=str, default="day_2", help="Target reference session (default: day_2)")
    parser.add_argument(
        "--matlab_exe",
        type=str,
        default=r"C:\Program Files\MATLAB\R2022b\bin\matlab.exe",
        help="Path to MATLAB R2022b executable"
    )
    parser.add_argument("--all_rois", action="store_true", help="Include all ROIs instead of only curated cells (iscell==1)")
    parser.add_argument("--iou_thresh", type=float, default=0.25, help="Minimum IoU overlap for matching cells (default: 0.25)")
    parser.add_argument("--max_dist", type=float, default=12.0, help="Maximum centroid distance in pixels (default: 12.0)")
    parser.add_argument("--skip_matlab", action="store_true", help="Skip Stage 1 & 2 if PatchWarp results already exist")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mouse_dir = repo_root / "data" / args.mouse
    prep_dir = mouse_dir / "patchwarp_prep"
    results_dir = mouse_dir / "patchwarp_results"
    matlab_script_dir = Path(__file__).resolve().parent / "matlab"

    if not mouse_dir.exists():
        raise FileNotFoundError(f"Mouse directory not found: {mouse_dir}")

    if args.target not in args.days:
        raise ValueError(f"Target session '{args.target}' must be one of the --days: {args.days}")

    prep_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_matlab:
        # ---------------------------------------------------------
        # STAGE 1: Extract Suite2p Templates & Masks to .mat
        # ---------------------------------------------------------
        print("====================================================")
        print(f" STAGE 1: Extracting Suite2p Data for {args.mouse}")
        print("====================================================")
        for day in args.days:
            day_dir = mouse_dir / day
            process_suite2p_session(day_dir, prep_dir, session_name=day, only_cells=not args.all_rois)

        # ---------------------------------------------------------
        # STAGE 2: Execute PatchWarp in MATLAB R2022b
        # ---------------------------------------------------------
        print("\n====================================================")
        print(" STAGE 2: Running PatchWarp in MATLAB R2022b")
        print("====================================================")
        days_cell_str = "{" + ", ".join([f"'{d}'" for d in args.days]) + "}"
        matlab_cmd = (
            f"repo_root='{repo_root}'; "
            f"mouse_id='{args.mouse}'; "
            f"session_names={days_cell_str}; "
            f"target_name='{args.target}'; "
            f"cd('{matlab_script_dir}'); "
            f"run_patchwarp_across;"
        )

        cmd = [args.matlab_exe, "-batch", matlab_cmd]
        print("Launching MATLAB R2022b...\n")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print("\n[ERROR] MATLAB PatchWarp execution failed.")
            sys.exit(result.returncode)

    # ---------------------------------------------------------
    # STAGE 3: Longitudinal ROI Matching & Trace Extraction
    # ---------------------------------------------------------
    print("\n====================================================")
    print(" STAGE 3: Matching Longitudinal Neurons & Traces")
    print("====================================================")
    match_across_sessions(
        mouse_dir=mouse_dir,
        days=args.days,
        target_day=args.target,
        iou_thresh=args.iou_thresh,
        max_dist=args.max_dist
    )

    print("\n====================================================")
    print(f" PIPELINE COMPLETE FOR {args.mouse}!")
    print(f" Prep files:    {prep_dir}")
    print(f" Final outputs: {results_dir}")
    print("====================================================")


if __name__ == "__main__":
    main()