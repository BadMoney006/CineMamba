# CineMamba

Official implementation of **CineMamba: Cardiac-Phase-Conditioned Temporal State Spaces for Accelerated Cine MRI**.

CineMamba combines a multi-scale spatial Mamba regularizer with cardiac-phase-conditioned temporal state evolution (CP-Mamba) and motion-driven spatial modulation (MDSM) in an eight-cascade reconstruction network. The full model is denoted **C4**.

## Installation

Install a CUDA-compatible PyTorch build and the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data preparation

The CMRxRecon2023 Cine loader expects `AccFactor04`, `AccFactor08`, and `AccFactor10` directories containing `P*/cine_sax.mat` and `P*/cine_lax.mat`, with `FullSample` as a sibling directory.

Estimate cardiac phase from the acquired k-space for each acceleration factor:

```bash
python -m cinemamba.phase \
  --input-root /path/to/CMRxRecon2023/Cine/TrainingSet/AccFactor04 \
  --acc-factor 04 \
  --output-root /path/to/phase
```

Repeat with `08` and `10` for the other acceleration factors. Phase arrays are saved under `<output-root>/<acc-factor>/<subject>/`.

## Training

```bash
python train.py \
  --run-name c4_2023 \
  --root-path /path/to/AccFactor04 /path/to/AccFactor08 /path/to/AccFactor10 \
  --phase-root /path/to/phase \
  --variant C4
```

The training entry point also supports the `C1`, `C1b`, `C2`, and `C3` ablations via `--variant`. Checkpoints and logs are written to `runs/<run-name>/` by default.

## Acknowledgments

This implementation uses components from [PromptMR](https://github.com/hellopipu/PromptMR) and [MambaIR](https://github.com/csguoh/MambaIR). See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution and licenses.
