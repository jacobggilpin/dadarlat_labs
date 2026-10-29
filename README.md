[![DOI](https://zenodo.org/badge/1436417760.svg)](https://doi.org/10.5281/zenodo.23768836)

# Dadarlat Labs Monorepo

Repository for 2-photon calcium imaging pipelines, Scanbox acquisition utilities, longitudinal registration, and intracortical microstimulation (ICMS) neural response analysis.

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

## Pipeline Architecture & Workflow

The repository is modularized into sequential pipelines to cleanly separate data ingestion, spatial alignment, and downstream neuroscience analysis:

```text
dadarlat_labs/
└── pipelines/
    ├── core/                         # Shared mathematical/utility functions
    ├── preprocessing/                # Stage 0: Scanbox .sbx -> .h5 conversion
    └── multi_day_alignment/          
        ├── run_suite2p_patchwarp.py  # Stage 1: Alignment Orchestrator
        ├── suite2p_patchwarp/        # Backend Python/MATLAB registration engine
        └── analysis/                 # Stage 2: Post-alignment visualizers & metrics
