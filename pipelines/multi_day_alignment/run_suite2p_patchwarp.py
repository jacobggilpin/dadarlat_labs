"""
run_suite2p_patchwarp.py
Streamlined data primer for longitudinal alignment.
Executes:
  Stage 1: Suite2p template & mask extraction (Python)
  Stage 2: PatchWarp across-session affine registration (MATLAB R2022b)
  Stage 3: Longitudinal ROI matching & trace compilation (Python)

This script ONLY aligns and prepares the data. Downstream analysis is handled separately.

Example usage:
    python pipelines\multi_day_alignment\run_suite2p_patchwarp.py --mouse mouse_6416_1R1L --days day_1 day_2 day_3 --target day_2 --run_name run_01_provided
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add the nested scripts folder to the path
script_dir = Path(__file__).resolve().parent / "suite2p_patchwarp" / "scripts"
sys.path.append(str(script_dir))

from suite2p_to_patchwarp import process_suite2p_session
from match_longitudinal_rois import match_across_sessions


def main():
    parser = argparse.ArgumentParser(description="Multi-Day Suite2p to PatchWarp Alignment Pipeline")
    parser.add_argument("--mouse", type=str, required=True, help="Mouse folder name inside data/ (e.g. mouse_6416_1R1L)")
    parser.add_argument("--days", nargs="+", default=["day_1", "day_2", "day_3"], help="Subfolder names for each day")
    parser.add_argument("--target", type=str, default="day_2", help="Target reference session (default: day_2)")
    parser.add_argument("--run_name", type=str, required=True, help="Target run folder (e.g., run_01_provided)")
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

    # From multi_day_alignment/ to pipelines/ to dadarlat_labs/
    repo_root = Path(__file__).resolve().parents[2]
    mouse_dir = repo_root / "data" / args.mouse
    
    run_out_dir = mouse_dir / "multi_day_results" / args.run_name
    prep_dir = run_out_dir / "patchwarp_prep"
    results_dir = run_out_dir / "patchwarp_results"
    matlab_script_dir = Path(__file__).resolve().parent / "suite2p_patchwarp" / "matlab"

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
            # FIX: Removed the extra "runs" folder to match user's actual directory structure
            day_dir = mouse_dir / day / args.run_name
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
            f"run_name='{args.run_name}'; "
            f"input_dir='{prep_dir}'; "
            f"output_dir='{results_dir}'; "
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
        run_name=args.run_name,
        iou_thresh=args.iou_thresh,
        max_dist=args.max_dist
    )

    print("\n====================================================")
    print(f" ALIGNMENT COMPLETE FOR {args.mouse}!")
    print(f" Data is primed and ready for longitudinal analysis.")
    print(f" Outputs saved to: {results_dir}")
    print("====================================================")


if __name__ == "__main__":
    main()
