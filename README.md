# Dadarlat Labs Monorepo

Repository for 2-photon calcium imaging pipelines, Scanbox acquisition utilities, and intracortical microstimulation (ICMS) analysis.

---

## Documentation & Literature Reviews

* **[Review of 2-Photon Microscopy in vivo Calcium Imaging Studies](docs/2P_Calcium_Imaging_Lit_Review.pdf)**
  * **Author:** Jake G. Gilpin (Dadarlat Labs, Weldon School of Biomedical Engineering, Purdue University)
  * **Topics Covered:** 
    * Spatiotemporal dynamics of excitatory vs. inhibitory cortical recruitment under ICMS.
    * Spike deconvolution algorithms (OASIS vs. CNNs vs. NND).
    * Suite2p processing architecture and PatchWarp non-rigid distortion correction.
    * Methodological confounds: ATP-depleted Ca2+ clearance failure, hemodynamic absorption, vascular tissue warping, and inter-trial carry-over.
    * Scanbox volumetric ETL imaging and closed-loop Z-drift stabilization.

---

## Suite2p to PatchWarp Pipeline

This pipeline automatically extracts Suite2p templates and ROI masks, bridges them to MATLAB, and runs PatchWarp's across-session affine registration to correct longitudinal tissue drift.

### 1. Prerequisites
* **Python:** Miniforge3 `suite2p` conda environment (see `pipelines/suite2p_patchwarp/environment.yml`).
* **MATLAB:** MATLAB R2022b (requires Image Processing, Parallel Computing, Signal Processing, and Statistics toolboxes).
* **PatchWarp:** Ensure the submodule is cloned into `external/PatchWarp`.

### 2. Directory Structure
Ensure your raw data is organized inside the `data/` folder (which is ignored by Git to prevent tracking massive binaries):
```text
data/
└── mouse_6416_1R1L/
    ├── day_1/suite2p/plane0/
    ├── day_2/suite2p/plane0/
    └── day_3/suite2p/plane0/