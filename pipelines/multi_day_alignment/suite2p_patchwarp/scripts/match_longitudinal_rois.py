import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def load_hdf5_masks_and_indices(h5_group, mask_key, idx_key):
    raw_masks = np.array(h5_group[mask_key])
    if raw_masks.ndim == 3:
        masks = np.transpose(raw_masks, (0, 2, 1)).astype(bool)
    elif raw_masks.ndim == 2:
        masks = np.transpose(raw_masks, (1, 0))[None, :, :].astype(bool)
    else:
        raise ValueError(f"Unexpected mask dimensions: {raw_masks.shape}")

    indices = np.array(h5_group[idx_key]).flatten().astype(int) - 1
    return masks, indices

def compute_centroids(masks):
    N = masks.shape[0]
    centroids = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        ys, xs = np.where(masks[i])
        if len(ys) > 0:
            centroids[i, 0] = np.mean(ys)
            centroids[i, 1] = np.mean(xs)
        else:
            centroids[i] = [-1000.0, -1000.0]
    return centroids

def match_session_pair(target_masks, source_masks, iou_thresh=0.25, max_dist=12.0):
    Nt = target_masks.shape[0]
    Ns = source_masks.shape[0]

    t_flat = target_masks.reshape(Nt, -1).astype(np.float32)
    s_flat = source_masks.reshape(Ns, -1).astype(np.float32)

    intersection = t_flat @ s_flat.T
    t_areas = t_flat.sum(axis=1)[:, None]
    s_areas = s_flat.sum(axis=1)[None, :]
    union = t_areas + s_areas - intersection

    iou_matrix = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    t_cent = compute_centroids(target_masks)
    s_cent = compute_centroids(source_masks)
    dist_matrix = cdist(t_cent, s_cent, metric="euclidean")

    cost_matrix = 1.0 - iou_matrix
    invalid_mask = (iou_matrix < iou_thresh) | (dist_matrix > max_dist)
    cost_matrix[invalid_mask] = 1e6

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1e5:
            matches[r] = (c, float(iou_matrix[r, c]), float(dist_matrix[r, c]))

    return matches

def match_across_sessions(
    mouse_dir: Path,
    days: list,
    target_day: str,
    run_name: str,
    results_dir: Path,
    iou_thresh: float = 0.25,
    max_dist: float = 12.0
):
    mat_path = results_dir / "across_sessions_patchwarp_results.mat"

    if not mat_path.exists():
        raise FileNotFoundError(f"PatchWarp output file not found: {mat_path}")

    print(f"\nLoading warped masks from: {mat_path.name}")
    source_days = [d for d in days if d != target_day]
    session_data = {}

    with h5py.File(mat_path, "r") as f:
        target_masks, target_indices = load_hdf5_masks_and_indices(f, "target_masks", "target_indices")
        target_align_img = np.transpose(np.array(f["target_alignImg"]), (1, 0))
        session_data[target_day] = {
            "masks": target_masks,
            "suite2p_indices": target_indices,
            "mean_img": target_align_img
        }

        for s_day in source_days:
            grp = f[s_day]
            s_masks, s_indices = load_hdf5_masks_and_indices(grp, "warped_masks", "roi_indices")
            s_align_img = np.transpose(np.array(grp["warped_alignImg"]), (1, 0))
            session_data[s_day] = {
                "masks": s_masks,
                "suite2p_indices": s_indices,
                "mean_img": s_align_img
            }

    Nt = len(target_indices)
    registry_rows = []
    for t_idx in range(Nt):
        row = {
            f"{target_day}_local_idx": t_idx,
            f"{target_day}_suite2p_id": int(target_indices[t_idx])
        }
        registry_rows.append(row)

    df = pd.DataFrame(registry_rows)

    for s_day in source_days:
        print(f"Matching {s_day} -> {target_day} (IoU >= {iou_thresh}, Dist <= {max_dist} px)...")
        matches = match_session_pair(
            target_masks,
            session_data[s_day]["masks"],
            iou_thresh=iou_thresh,
            max_dist=max_dist
        )
        s_indices = session_data[s_day]["suite2p_indices"]

        s_local_col = []
        s_s2p_col = []
        iou_col = []
        dist_col = []

        for t_idx in range(Nt):
            if t_idx in matches:
                c_idx, iou_val, dist_val = matches[t_idx]
                s_local_col.append(c_idx)
                s_s2p_col.append(int(s_indices[c_idx]))
                iou_col.append(round(iou_val, 4))
                dist_col.append(round(dist_val, 2))
            else:
                s_local_col.append(-1)
                s_s2p_col.append(-1)
                iou_col.append(np.nan)
                dist_col.append(np.nan)

        df[f"{s_day}_local_idx"] = s_local_col
        df[f"{s_day}_suite2p_id"] = s_s2p_col
        df[f"{s_day}_iou"] = iou_col
        df[f"{s_day}_dist_px"] = dist_col

    s2p_cols = [f"{d}_suite2p_id" for d in days]
    tracked_all_mask = (df[s2p_cols] != -1).all(axis=1)
    df_tracked_all = df[tracked_all_mask].reset_index(drop=True)
    n_tracked = len(df_tracked_all)
    print(f"\n✓ Found {n_tracked} neurons consistently tracked across ALL {len(days)} sessions: {days}")

    csv_all_path = results_dir / "cell_registry_all_target_cells.csv"
    csv_tracked_path = results_dir / "cell_registry_tracked_all_days.csv"
    df.to_csv(csv_all_path, index=False)
    df_tracked_all.to_csv(csv_tracked_path, index=False)

    compiled_traces = {
        "tracked_registry": df_tracked_all.to_dict(orient="list"),
        "days": days,
        "target_day": target_day,
        "sessions": {}
    }

    for d in days:
        # PULL FROM RAW RUN_NAME
        plane0 = mouse_dir / d / run_name / "suite2p" / "plane0"
        F = np.load(plane0 / "F.npy", allow_pickle=True)
        Fneu = np.load(plane0 / "Fneu.npy", allow_pickle=True)
        spks = np.load(plane0 / "spks.npy", allow_pickle=True)
        dFF = F - 0.7 * Fneu

        matched_ids = df_tracked_all[f"{d}_suite2p_id"].values.astype(int)
        compiled_traces["sessions"][d] = {
            "suite2p_ids": matched_ids,
            "F": F[matched_ids, :] if n_tracked > 0 else np.empty((0, F.shape[1])),
            "Fneu": Fneu[matched_ids, :] if n_tracked > 0 else np.empty((0, Fneu.shape[1])),
            "dFF_corr": dFF[matched_ids, :] if n_tracked > 0 else np.empty((0, F.shape[1])),
            "spks": spks[matched_ids, :] if n_tracked > 0 else np.empty((0, spks.shape[1]))
        }

    traces_out_path = results_dir / "matched_longitudinal_traces.npy"
    np.save(traces_out_path, compiled_traces)

    colors = ["#00FF66", "#FF3399", "#00CCFF", "#FFCC00"]
    for i, d in enumerate(days):
        c = colors[i % len(colors)]
        fig, ax = plt.subplots(figsize=(6, 6))
        
        day_mean_img = session_data[d]["mean_img"]
        vmin, vmax = np.percentile(day_mean_img, [1, 99])
        ax.imshow(day_mean_img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off")
        
        masks_d = session_data[d]["masks"]
        local_indices = df_tracked_all[f"{d}_local_idx"].values.astype(int)
        if len(local_indices) > 0:
            ax.contour(np.max(masks_d[local_indices], axis=0), levels=[0.5], colors=[c], linewidths=1.0)
            
        ax.set_title(f"{d} Tracked (n={n_tracked})", fontsize=12, color=c)
        plt.tight_layout()
        plt.savefig(results_dir / f"qc_matched_rois_{d}.png", dpi=200)
        plt.close(fig)

    fig_panels, axes = plt.subplots(1, len(days), figsize=(5 * len(days), 5))
    if len(days) == 1: axes = [axes]
        
    for i, d in enumerate(days):
        ax = axes[i]
        c = colors[i % len(colors)]
        day_mean_img = session_data[d]["mean_img"]
        vmin, vmax = np.percentile(day_mean_img, [1, 99])
        ax.imshow(day_mean_img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off")
        masks_d = session_data[d]["masks"]
        local_indices = df_tracked_all[f"{d}_local_idx"].values.astype(int)
        if len(local_indices) > 0:
            ax.contour(np.max(masks_d[local_indices], axis=0), levels=[0.5], colors=[c], linewidths=1.0)
        ax.set_title(f"{d} (n={n_tracked})", fontsize=12, color=c)

    plt.tight_layout()
    plt.savefig(results_dir / "matched_rois_panels.png", dpi=200)
    plt.close(fig_panels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_dir", type=str, required=True)
    parser.add_argument("--days", nargs="+", default=["day_1", "day_2", "day_3"])
    parser.add_argument("--target", type=str, default="day_2")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--iou_thresh", type=float, default=0.25)
    parser.add_argument("--max_dist", type=float, default=12.0)
    args = parser.parse_args()

    match_across_sessions(
        Path(args.mouse_dir).resolve(),
        args.days,
        args.target,
        args.run_name,
        Path(args.results_dir).resolve(),
        iou_thresh=args.iou_thresh,
        max_dist=args.max_dist
    )