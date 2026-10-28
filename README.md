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
    * Methodological confounds: ATP-depleted $Ca^{2+}$ clearance failure, hemodynamic absorption, vascular tissue warping, and inter-trial carry-over.
    * Scanbox volumetric ETL imaging and closed-loop Z-drift stabilization.

---

## Pipelines & Tools

* **`pipelines/suite2p_patchwarp/`**: End-to-end pipeline for converting Scanbox `.sbx` files to HDF5, running Suite2p motion correction & ROI extraction, and applying PatchWarp across-session alignment.
* **`shared_utils/`**: Reusable data conversion and signal processing scripts.
