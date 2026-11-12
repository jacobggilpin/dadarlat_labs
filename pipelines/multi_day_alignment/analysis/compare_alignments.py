"""
Usage (One-Line Command from Repository Root):
    python pipelines\multi_day_alignment\analysis\compare_alignments.py --mouse_dir data\mouse_6416_1R1L --run_name run_01_spa --target_day day_2
"""

import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Compare tracked cells across alignment settings.")
    parser.add_argument("--mouse_dir", type=str, required=True, help="Path to mouse dir (e.g. data/mouse_6416_1R1L)")
    parser.add_argument("--run_name", type=str, required=True, help="Base run name (e.g. run_00_given)")
    parser.add_argument("--target_day", type=str, default="day_2", help="The target session name used in the pipeline")
    args = parser.parse_args()

    results_base = Path(args.mouse_dir) / "multi_day_results"
    
    settings = ["meanImg", "meanImgE", "max_proj"]
    tracked_sets = {}

    for s in settings:
        csv_path = results_base / f"{args.run_name}_{s}" / "patchwarp_results" / "cell_registry_tracked_all_days.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # We use the target day's Suite2p ID as the universal identifier for the cell
            ids = set(df[f"{args.target_day}_suite2p_id"].astype(int))
            tracked_sets[s] = ids
            print(f"[{s}] Total tracked cells: {len(ids)}")
        else:
            print(f"Warning: Could not find {csv_path.name} for {s}")
            tracked_sets[s] = set()

    if not tracked_sets:
        return

    s1, s2, s3 = settings
    set1, set2, set3 = tracked_sets[s1], tracked_sets[s2], tracked_sets[s3]

    print(f"\n--- Alignment Comparison Analysis ---")
    
    # Cells tracked by ALL methods
    intersection_all = set1 & set2 & set3
    print(f"Robust cells tracked by ALL 3 methods: {len(intersection_all)}")

    # Unique to specific methods
    unique_meanImg = set1 - (set2 | set3)
    unique_meanImgE = set2 - (set1 | set3)
    unique_max_proj = set3 - (set1 | set2)

    print(f"\nUnique to {s1} only: {len(unique_meanImg)}")
    if unique_meanImg: print(f"  Target Day Suite2p IDs: {sorted(list(unique_meanImg))}")

    print(f"Unique to {s2} only: {len(unique_meanImgE)}")
    if unique_meanImgE: print(f"  Target Day Suite2p IDs: {sorted(list(unique_meanImgE))}")

    print(f"Unique to {s3} only: {len(unique_max_proj)}")
    if unique_max_proj: print(f"  Target Day Suite2p IDs: {sorted(list(unique_max_proj))}")

if __name__ == "__main__":
    main()