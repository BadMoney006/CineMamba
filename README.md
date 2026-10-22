# CineMamba

Research code for **CineMamba: Cardiac-Phase-Conditioned Temporal State Spaces for Accelerated Cine MRI**.

The `C4` configuration is the complete eight-cascade model: spatial Mamba processing in a multiscale regularizer, seven cardiac-phase-conditioned temporal CP-Mamba blocks, and seven motion-driven spatial modulation (MDSM) heads. Time is preserved as a complete ten-frame cardiac cycle. The MDSM motion map uses all ten cyclic magnitude differences, including the final-to-first transition.

## Scope

The experiments reported in the paper were run and validated in a remote compute environment. This repository provides the CMRxRecon2023 Cine data loader, ACS-derived phase precomputation, model, training entry point, and validation metrics. CMRxRecon datasets, patient data, masks, and pretrained weights are not redistributed here. The CMRxRecon2025 preprocessing/evaluation pipeline is not included in this release.

## Setup

Use a CUDA-capable environment supported by your PyTorch and `mamba-ssm` versions. Install PyTorch for your CUDA version first, then install `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Data and phase

The loader expects one or more `AccFactorNN` directories with `P*/cine_sax.mat` and `P*/cine_lax.mat`; the corresponding `FullSample` directory is a sibling. Full-sample data are used **only as supervised targets**, never in the cardiac phase estimator. Keep the dataset outside this repository.

Generate phase from each acceleration factor's *acquired undersampled* k-space. For example:

```bash
python -m cinemamba.phase --input-root /data/CMRxRecon2023/Cine/TrainingSet/AccFactor04 --acc-factor 04 --output-root /data/phase --kc 8
python -m cinemamba.phase --input-root /data/CMRxRecon2023/Cine/TrainingSet/AccFactor08 --acc-factor 08 --output-root /data/phase --kc 8
python -m cinemamba.phase --input-root /data/CMRxRecon2023/Cine/TrainingSet/AccFactor10 --acc-factor 10 --output-root /data/phase --kc 8
```

Phase files are written as `/data/phase/<acc_factor>/<subject>/<view>_slice<index>.npy`. Check that the 8×8 ACS region is acquired in every relevant mask before relying on these signals.

## Training

```bash
python train.py \
  --run-name c4_2023 \
  --root-path /data/CMRxRecon2023/Cine/TrainingSet/AccFactor04 /data/CMRxRecon2023/Cine/TrainingSet/AccFactor08 /data/CMRxRecon2023/Cine/TrainingSet/AccFactor10 \
  --phase-root /data/phase \
  --variant C4
```

`--variant` also accepts `C1`, `C1b`, `C2`, and `C3` for ablations. `C2` uses uniform `t/T`; `C3` and `C4` require ACS-derived phase files. The training script holds out subjects from the supplied roots for local validation; do not equate those scores with the paper's test-set table. It uses a center-frame SSIM loss for each ten-frame cyclic input and saves outputs under `runs/`.

## Third-party code

`cinemamba/promptmr.py` contains PromptMR components by Bingyu Xin; `cinemamba/mambair.py` contains adapted MambaIR components. Their provenance and license terms are in [THIRD_PARTY.md](THIRD_PARTY.md) and `LICENSES/`. No license is granted here for the project's original code pending the authors' choice of a repository license.
