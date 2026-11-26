"""
run_suite2p_patchwarp.py
Streamlined data primer for longitudinal alignment.
Executes the pipeline iteratively across multiple align_img settings to compare outputs.

Usage (One-Line Command from Repository Root):
    python pipelines\multi_day_alignment\run_suite2p_patchwarp.py --mouse mouse_6416_1R1L --days day_1 day_2 day_3 --target day_2 --run_name run_01_spa
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
    parser.add_argument("--mouse", type=str, required=True, help="Mouse folder name inside data/")
    parser.add_argument("--days", nargs="+", default=["day_1", "day_2", "day_3"], help="Subfolder names for each day")
    parser.add_argument("--target", type=str, default="day_2", help="Target reference session")
    parser.add_argument("--run_name", type=str, required=True, help="Input run folder (e.g., run_00_given)")
    parser.add_argument(
        "--align_imgs",
        nargs="+",
        default=["meanImg", "meanImgE", "max_proj"],
        help="List of align settings to loop through."
    )
    parser.add_argument(
        "--matlab_exe",
        type=str,
        default=r"C:\Program Files\MATLAB\R2022b\bin\matlab.exe",
        help="Path to MATLAB executable"
    )
    parser.add_argument("--all_rois", action="store_true", help="Include all ROIs")
    parser.add_argument("--iou_thresh", type=float, default=0.25, help="Minimum IoU overlap")
    parser.add_argument("--max_dist", type=float, default=12.0, help="Maximum centroid distance")
    parser.add_argument("--skip_matlab", action="store_true", help="Skip Stage 1 & 2")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mouse_dir = repo_root / "data" / args.mouse
    matlab_script_dir = Path(__file__).resolve().parent / "suite2p_patchwarp" / "matlab"

    if not mouse_dir.exists():
        raise FileNotFoundError(f"Mouse directory not found: {mouse_dir}")

    for align_setting in args.align_imgs:
        print(f"\n{'='*60}")
        print(f" STARTING PIPELINE FOR ALIGN_IMG: {align_setting}")
        print(f"{'='*60}")

        # Create isolated output directories based on the setting
        specific_run_name = f"{args.run_name}_{align_setting}"
        run_out_dir = mouse_dir / "multi_day_results" / specific_run_name
        prep_dir = run_out_dir / "patchwarp_prep"
        results_dir = run_out_dir / "patchwarp_results"
        
        prep_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_matlab:
            print("\n--- STAGE 1: Extracting Suite2p Data ---")
            for day in args.days:
                # Always pull input data from the base run_name folder
                day_dir = mouse_dir / day / args.run_name
                process_suite2p_session(
                    day_dir, 
                    prep_dir, 
                    session_name=day, 
                    only_cells=not args.all_rois, 
                    align_img_key=align_setting
                )

            print("\n--- STAGE 2: Running PatchWarp in MATLAB R2022b ---")
            days_cell_str = "{" + ", ".join([f"'{d}'" for d in args.days]) + "}"
            matlab_cmd = (
                f"repo_root='{repo_root}'; "
                f"mouse_id='{args.mouse}'; "
                f"run_name='{specific_run_name}'; "
                f"input_dir='{prep_dir}'; "
                f"output_dir='{results_dir}'; "
                f"session_names={days_cell_str}; "
                f"target_name='{args.target}'; "
                f"cd('{matlab_script_dir}'); "
                f"run_patchwarp_across;"
            )

            cmd = [args.matlab_exe, "-batch", matlab_cmd]
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f"\n[ERROR] MATLAB failed for {align_setting}. Skipping to next.")
                continue

        print("\n--- STAGE 3: Matching Longitudinal Neurons & Traces ---")
        match_across_sessions(
            mouse_dir=mouse_dir,
            days=args.days,
            target_day=args.target,
            run_name=args.run_name,
            results_dir=results_dir,
            iou_thresh=args.iou_thresh,
            max_dist=args.max_dist
        )
        print(f"\n✓ Completed processing for {align_setting}. Outputs saved to: {results_dir.name}")

    print("\n====================================================")
    print(" ALL ALIGNMENT SETTINGS PROCESSED SUCCESSFULLY!")
    print("====================================================")

if __name__ == "__main__":
    main()