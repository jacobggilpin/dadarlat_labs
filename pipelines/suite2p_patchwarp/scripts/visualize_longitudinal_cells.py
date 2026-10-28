"""
visualize_longitudinal_cells.py
Interactive viewer and longitudinal analysis tool for Suite2p + PatchWarp tracked neurons.

Usage (One-Line Command from Repository Root):
    python pipelines\\suite2p_patchwarp\\scripts\\visualize_longitudinal_cells.py --mouse mouse_6416_1R1L --fs 15.625

Key Optimizations:
  - Artist data-updating (set_data / set_ydata) instead of ax.clear() for <15ms navigation
    while preserving interactive zoom/pan synchronization between dF/F and OASIS spikes.
  - Biophysically accurate neuropil-corrected dF/F0 normalization (preventing near-zero division).
  - Symmetric boundary padding so edge-of-FOV neurons remain centered in crops.
  - Multi-contour overlay on the target session to directly inspect cross-day ROI overlap.
"""

import argparse
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import pandas as pd


def compute_robust_dff(F: np.ndarray, Fneu: np.ndarray, r: float = 0.7, pct: float = 20.0):
    """
    Computes biophysically accurate neuropil-corrected Delta F / F0.
    
    Formula:
        F_corr  = F - r * Fneu
        F0_corr = percentile(F_corr, pct)
        F0_raw  = maximum(percentile(F, pct), 1.0)
        dFF     = (F_corr - F0_corr) / F0_raw
    """
    F_corr = F.astype(np.float32) - r * Fneu.astype(np.float32)
    f0_corr = np.percentile(F_corr, pct, axis=1, keepdims=True)
    f0_raw = np.maximum(np.percentile(F.astype(np.float32), pct, axis=1, keepdims=True), 1.0)
    dff = (F_corr - f0_corr) / f0_raw
    return dff, f0_raw.squeeze(axis=1)


def load_h5_masks(dataset) -> np.ndarray:
    """Safely loads and transposes MATLAB v7.3 HDF5 3D mask arrays to [N, Ly, Lx]."""
    arr = np.array(dataset)
    if arr.ndim == 3:
        return np.transpose(arr, (0, 2, 1)).astype(bool)
    elif arr.ndim == 2:
        return np.transpose(arr, (1, 0))[None, :, :].astype(bool)
    raise ValueError(f"Unexpected HDF5 mask shape: {arr.shape}")


class LongitudinalViewer:
    def __init__(self, mouse_dir: Path, crop_radius: int = 35, fs: float = 15.625):
        self.mouse_dir = mouse_dir
        self.crop_radius = crop_radius
        self.fs = fs  # Scanbox frame rate in Hz (default: 15.625 Hz)
        self.results_dir = mouse_dir / "patchwarp_results"

        csv_path = self.results_dir / "cell_registry_tracked_all_days.csv"
        traces_path = self.results_dir / "matched_longitudinal_traces.npy"
        mat_path = self.results_dir / "across_sessions_patchwarp_results.mat"

        for p in (csv_path, traces_path, mat_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing required pipeline output: {p}")

        self.df = pd.read_csv(csv_path)
        self.traces_data = np.load(traces_path, allow_pickle=True).item()
        self.days = list(self.traces_data["days"])
        self.target_day = self.traces_data["target_day"]
        self.n_cells = len(self.df)
        self.current_idx = 0

        if self.n_cells == 0:
            raise RuntimeError("No neurons were tracked across all sessions (n_cells == 0).")

        print(f"Loaded {self.n_cells} tracked neurons across sessions: {self.days} (fs = {self.fs} Hz)")

        # 1. Load Reference Images & Masks from MATLAB v7.3 HDF5
        self.mean_imgs = {}
        self.masks = {}
        r = self.crop_radius

        with h5py.File(mat_path, "r") as f:
            # Target session
            t_img = np.transpose(np.array(f["target_meanImg"]), (1, 0)).astype(np.float32)
            t_masks = load_h5_masks(f["target_masks"])
            self.mean_imgs[self.target_day] = np.pad(t_img, r, mode="edge")
            self.masks[self.target_day] = np.pad(t_masks, ((0, 0), (r, r), (r, r)), mode="constant")

            # Source sessions
            for d in self.days:
                if d != self.target_day:
                    s_img = np.transpose(np.array(f[d]["warped_meanImg"]), (1, 0)).astype(np.float32)
                    s_masks = load_h5_masks(f[d]["warped_masks"])
                    self.mean_imgs[d] = np.pad(s_img, r, mode="edge")
                    self.masks[d] = np.pad(s_masks, ((0, 0), (r, r), (r, r)), mode="constant")

        # 2. Precompute Normalized dF/F0, Spikes, and Time Vectors
        self.dff_data = {}
        self.f0_data = {}
        self.spks_data = {}
        self.time_vecs = {}

        for d in self.days:
            sess = self.traces_data["sessions"][d]
            F = np.atleast_2d(sess["F"])
            Fneu = np.atleast_2d(sess["Fneu"])
            spks = np.atleast_2d(sess["spks"]).astype(np.float32)

            dff, f0 = compute_robust_dff(F, Fneu)
            self.dff_data[d] = dff
            self.f0_data[d] = f0
            self.spks_data[d] = spks
            self.time_vecs[d] = np.arange(dff.shape[1], dtype=np.float32) / self.fs

        # High-contrast color palette (Green, Magenta, Cyan, Gold)
        self.day_colors = ["#00E676", "#FF4081", "#00E5FF", "#FFD740"]
        self.contour_collections = [[] for _ in self.days]

        self._init_figure()
        self.update_plot(0)

    def _init_figure(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 9), facecolor="#121212")
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(
                f"Longitudinal Cell Explorer — {self.mouse_dir.name}"
            )

        n_days = len(self.days)
        grid = plt.GridSpec(
            3, n_days,
            height_ratios=[1.3, 1.0, 0.45],
            hspace=0.38, wspace=0.22,
            left=0.06, right=0.97, top=0.90, bottom=0.14
        )

        self.ax_imgs = []
        self.ax_traces = []
        self.ax_spks = []

        self.img_artists = []
        self.trace_lines = []
        self.spk_lines = []

        crop_dim = 2 * self.crop_radius
        dummy_crop = np.zeros((crop_dim, crop_dim), dtype=np.float32)

        for i, d in enumerate(self.days):
            col_color = self.day_colors[i % len(self.day_colors)]
            t_vec = self.time_vecs[d]
            z_vec = np.zeros_like(t_vec)

            # 1. Image Crop Axes
            ax_im = self.fig.add_subplot(grid[0, i])
            im_art = ax_im.imshow(dummy_crop, cmap="gray", origin="upper", interpolation="nearest")
            ax_im.axhline(self.crop_radius, color="white", ls=":", lw=0.6, alpha=0.4)
            ax_im.axvline(self.crop_radius, color="white", ls=":", lw=0.6, alpha=0.4)
            ax_im.axis("off")
            self.ax_imgs.append(ax_im)
            self.img_artists.append(im_art)

            # 2. Delta F/F0 Trace Axes
            ax_tr = self.fig.add_subplot(grid[1, i])
            ax_tr.set_facecolor("#1A1A1A")
            (tr_line,) = ax_tr.plot(t_vec, z_vec, color=col_color, lw=0.85, alpha=0.95)
            ax_tr.axhline(0, color="#888888", ls="--", lw=0.6, alpha=0.5)
            if i == 0:
                ax_tr.set_ylabel(r"$\Delta F / F_0$ (Neuropil Corr.)", fontsize=10)
            ax_tr.grid(True, linestyle=":", color="#333333", alpha=0.6)
            ax_tr.set_xlim(0, t_vec[-1])
            self.ax_traces.append(ax_tr)
            self.trace_lines.append(tr_line)

            # 3. Deconvolved Spikes Axes (Shared X-Axis for synchronized zoom)
            ax_sp = self.fig.add_subplot(grid[2, i], sharex=ax_tr)
            ax_sp.set_facecolor("#1A1A1A")
            (sp_line,) = ax_sp.plot(t_vec, z_vec, color="#B0BEC5", lw=0.75, alpha=0.9)
            if i == 0:
                ax_sp.set_ylabel("OASIS Spikes (a.u.)", fontsize=9)
            ax_sp.set_xlabel("Time (seconds)", fontsize=9)
            ax_sp.grid(True, linestyle=":", color="#333333", alpha=0.6)
            self.ax_spks.append(ax_sp)
            self.spk_lines.append(sp_line)

        # Controls: Slider & Prev/Next Buttons
        ax_slider = plt.axes([0.25, 0.04, 0.50, 0.03], facecolor="#2A2A2A")
        self.slider = Slider(
            ax_slider, "Neuron ", 0, self.n_cells - 1,
            valinit=0, valstep=1, valfmt="%d", color="#00E676"
        )
        self.slider.on_changed(lambda val: self.update_plot(int(val)))

        ax_prev = plt.axes([0.13, 0.04, 0.08, 0.035])
        self.btn_prev = Button(ax_prev, "◄ Prev [Left]", color="#2A2A2A", hovercolor="#444444")
        self.btn_prev.on_clicked(lambda _: self._step(-1))

        ax_next = plt.axes([0.79, 0.04, 0.08, 0.035])
        self.btn_next = Button(ax_next, "Next [Right] ►", color="#2A2A2A", hovercolor="#444444")
        self.btn_next.on_clicked(lambda _: self._step(1))

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _step(self, delta: int):
        new_idx = int(np.clip(self.current_idx + delta, 0, self.n_cells - 1))
        if new_idx != self.current_idx:
            self.slider.set_val(new_idx)

    def _on_key(self, event):
        if event.key in ("right", "up"):
            self._step(1)
        elif event.key in ("left", "down"):
            self._step(-1)

    def _clear_contours(self, axis_idx: int):
        for c_set in self.contour_collections[axis_idx]:
            try:
                c_set.remove()
            except Exception:
                for coll in getattr(c_set, "collections", []):
                    try:
                        coll.remove()
                    except Exception:
                        pass
        self.contour_collections[axis_idx].clear()

    def update_plot(self, cell_idx: int):
        self.current_idx = cell_idx
        row = self.df.iloc[cell_idx]
        r = self.crop_radius

        # Compute padded target centroid so crop is always [2*r, 2*r]
        t_loc_idx = int(row[f"{self.target_day}_local_idx"])
        target_mask_padded = self.masks[self.target_day][t_loc_idx]
        ys, xs = np.where(target_mask_padded)
        cy, cx = int(np.round(np.mean(ys))), int(np.round(np.mean(xs)))

        y_slice = slice(cy - r, cy + r)
        x_slice = slice(cx - r, cx + r)

        # Shared y-limit across all days for direct amplitude comparison
        all_max_dff = max(
            float(np.percentile(self.dff_data[d][cell_idx], 99.8)) for d in self.days
        )
        shared_ymax = max(0.8, all_max_dff * 1.25)

        for i, d in enumerate(self.days):
            col_color = self.day_colors[i % len(self.day_colors)]
            loc_idx = int(row[f"{d}_local_idx"])
            s2p_id = int(row[f"{d}_suite2p_id"])

            # 1. Update Image Crop & Contours
            crop = self.mean_imgs[d][y_slice, x_slice]
            vmin, vmax = np.percentile(crop, [3, 99])
            if vmax <= vmin:
                vmax = vmin + 1.0
            self.img_artists[i].set_data(crop)
            self.img_artists[i].set_clim(vmin, vmax)

            self._clear_contours(i)
            m_crop = self.masks[d][loc_idx][y_slice, x_slice].astype(np.float32)

            if d == self.target_day:
                # On the Target image, draw contours of ALL days to show direct overlap
                for j, d_other in enumerate(self.days):
                    other_loc = int(row[f"{d_other}_local_idx"])
                    other_mask = self.masks[d_other][other_loc][y_slice, x_slice].astype(np.float32)
                    lw = 1.8 if d_other == self.target_day else 1.0
                    ls = "solid" if d_other == self.target_day else "dashed"
                    c_set = self.ax_imgs[i].contour(
                        other_mask, levels=[0.5],
                        colors=[self.day_colors[j % len(self.day_colors)]],
                        linewidths=lw, linestyles=ls
                    )
                    self.contour_collections[i].append(c_set)
                subtext = "Target Ref (All Contours Overlaid)"
            else:
                c_set = self.ax_imgs[i].contour(
                    m_crop, levels=[0.5], colors=[col_color], linewidths=1.6
                )
                self.contour_collections[i].append(c_set)
                iou = row.get(f"{d}_iou", np.nan)
                dist = row.get(f"{d}_dist_px", np.nan)
                subtext = f"IoU: {iou:.2f}  |  Shift: {dist:.1f} px"

            self.ax_imgs[i].set_title(
                f"{d} — Suite2p ROI #{s2p_id}\n{subtext}",
                fontsize=10.5, color=col_color, fontweight="bold", pad=8
            )

            # 2. Update Delta F / F0 Trace
            dff_trace = self.dff_data[d][cell_idx]
            f0_val = self.f0_data[d][cell_idx]
            self.trace_lines[i].set_ydata(dff_trace)
            self.ax_traces[i].set_ylim(-0.25, shared_ymax)
            self.ax_traces[i].set_title(
                f"Resting Baseline $F_0$: {f0_val:.1f}  |  Peak $\\Delta F/F_0$: {np.max(dff_trace):.2f}",
                fontsize=9.5, color="#CCCCCC"
            )

            # 3. Update Deconvolved Spikes
            spks = self.spks_data[d][cell_idx]
            self.spk_lines[i].set_ydata(spks)
            spk_max = max(0.5, float(np.percentile(spks, 99.9)) * 1.25)
            self.ax_spks[i].set_ylim(0, spk_max)

        self.fig.suptitle(
            f"Tracked Longitudinal Neuron {cell_idx + 1} / {self.n_cells}   "
            f"[{self.mouse_dir.name}]",
            fontsize=14, fontweight="bold", color="white", y=0.97
        )
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="Optimized Multi-Day Longitudinal Cell Explorer")
    parser.add_argument("--mouse", type=str, default="mouse_6416_1R1L", help="Mouse folder inside data/")
    parser.add_argument("--crop", type=int, default=35, help="Crop radius around soma in pixels (default: 35)")
    parser.add_argument("--fs", type=float, default=15.625, help="Scanbox frame rate in Hz (default: 15.625)")
    args = parser.parse_args()

    # Resolve to repository root (dadarlat_labs/) from pipelines/suite2p_patchwarp/scripts/
    repo_root = Path(__file__).resolve().parents[3]
    mouse_dir = repo_root / "data" / args.mouse

    viewer = LongitudinalViewer(mouse_dir, crop_radius=args.crop, fs=args.fs)
    plt.show()


if __name__ == "__main__":
    main()
