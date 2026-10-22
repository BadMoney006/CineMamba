"""Training-free cardiac phase estimation from acquired CMRxRecon cine ACS.

Precomputes per-(acceleration, subject, view, slice) cardiac phase phi in [0, 1) from the
central low-frequency block of *undersampled* k-space (paper Section 3.2).
Never pass a FullSample directory here: doing so would leak reference data.

Output layout mirrors the raw dataset:

    <output_root>/<acc_factor>/<subject>/<view>_slice<slice_idx>.npy

Each file stores a float32 array of shape (T,) with phi_t ∈ [0, 1), where
phi = 0 approximately corresponds to end-diastole and phi ≈ 0.3 to peak-systole.

Design notes:
- Compute the phase independently for each acquired acceleration factor.
- PCA + Hilbert transform on the central K_c × K_c coil-averaged magnitude
  time-series. Robust against per-scan frame count.
- Precomputation is CPU-only, no GPU required.

Usage:
    python -m cinemamba.phase \
        --input-root /path/to/CMRxRecon2023/Cine/TrainingSet/AccFactor04 \
        --acc-factor 04 \
        --output-root /path/to/CMRxRecon2023/Cine/TrainingSet/Phase \
        --kc 8

Behavior:
- Skips (subject, view, slice) triples whose output .npy already exists (safe to
  re-run mid-execution).
- Writes a manifest phase_manifest.csv into --output-root listing every file
  produced, its shape, and success/skip/error status.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.signal import hilbert


# -------------------------------------------------------------------------
# Core algorithm — cardiac phase estimation
# -------------------------------------------------------------------------


def estimate_phase_from_kspace(
    kspace_slice: np.ndarray,
    kc: int = 8,
) -> np.ndarray:
    """Estimate cardiac phase phi ∈ [0, 1)^T from a single-slice cine k-space.

    Args:
        kspace_slice: acquired complex k-space with shape (T, C, H, W).
        kc: side length of the central low-frequency block used for phase
            estimation. Default 8 (=> 8×8 = 64 spatial points per frame).

    Returns:
        phi: float32 array of shape (T,) with values in [0, 1).
    """
    T, C, H, W = kspace_slice.shape
    if T < 4:
        # Too few frames for meaningful phase; return uniform t/T.
        return (np.arange(T, dtype=np.float32) / max(T, 1)) % 1.0

    # Step 1: central K_c × K_c block
    kc = min(kc, H, W)
    h_lo = (H - kc) // 2
    w_lo = (W - kc) // 2
    kspace_lo = kspace_slice[:, :, h_lo:h_lo + kc, w_lo:w_lo + kc]

    # Step 2: coil-averaged magnitude image at each frame
    # IFFT along the last two dims to move from k-space back to image domain.
    img_lo = np.fft.ifftshift(kspace_lo, axes=(-2, -1))
    img_lo = np.fft.ifft2(img_lo, axes=(-2, -1))
    img_lo = np.fft.fftshift(img_lo, axes=(-2, -1))
    mag = np.abs(img_lo).mean(axis=1)  # (T, kc, kc), average over coils

    # Step 3: flatten + stack → S ∈ R^{T × kc²}
    S = mag.reshape(T, -1)  # (T, kc²)

    # Center each spatial location across time (remove per-pixel mean).
    S = S - S.mean(axis=0, keepdims=True)

    # Step 4: first principal component → scalar temporal signal z ∈ R^T
    # SVD of the centered matrix; the first left singular vector times its
    # singular value gives the PC1 time-course.
    u, s, _vt = np.linalg.svd(S, full_matrices=False)
    z = u[:, 0] * s[0]  # (T,)

    # Step 5: Hilbert transform → analytic signal → instantaneous phase
    z_analytic = hilbert(z)
    inst_phase = np.angle(z_analytic)  # in [-pi, pi]

    # Normalize to [0, 1).
    phi = (inst_phase / (2.0 * np.pi)) % 1.0

    # Convention: shift so that phi = 0 aligns with end-diastole.
    # End-diastole ≈ frame with minimum |z| (heart most static ≈ lowest signal
    # variation projected onto PC1). Rotate phi so that phi[argmin |z|] = 0.
    t_ed = int(np.argmin(np.abs(z)))
    phi = (phi - phi[t_ed]) % 1.0

    return phi.astype(np.float32)


# -------------------------------------------------------------------------
# CMRxRecon 2023 file I/O helpers
# -------------------------------------------------------------------------


def read_acquired_kspace(mat_path: Path, acc_factor: str) -> np.ndarray | None:
    """Read one undersampled CMRxRecon 2023 cine_*.mat file.

    Returns:
        Complex array of shape (T, num_slices, C, H, W), or (T, num_slices, H, W)
        for single-coil data (in which case a coil axis is inserted).

        None if the file does not contain the requested acceleration factor.
    """
    with h5py.File(mat_path, "r") as f:
        keys = list(f.keys())

        target_key = None
        for k in (f"kspace_sub{acc_factor}", f"kspace_single_sub{acc_factor}"):
            if k in keys:
                target_key = k
                break
        if target_key is None:
            return None

        obj = f[target_key]
        try:
            real = obj["real"][:]
            imag = obj["imag"][:]
            arr = real + 1j * imag
        except (KeyError, TypeError, AttributeError):
            arr = obj[:]

    # Ensure coil dim exists
    if arr.ndim == 4:
        # (T, num_slices, H, W) — insert coil axis
        arr = arr[:, :, np.newaxis, :, :]
    elif arr.ndim != 5:
        raise ValueError(f"{mat_path}: unexpected acquired k-space shape {arr.shape}")
    return arr.astype(np.complex64)


def iter_cine_files(input_root: Path) -> Iterable[tuple[Path, str, str]]:
    """Yield (mat_path, subject, view) for acquired cine files."""
    for subj_dir in sorted(input_root.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith("P"):
            continue
        for view in ("sax", "lax"):
            mat_path = subj_dir / f"cine_{view}.mat"
            if mat_path.exists():
                yield mat_path, subj_dir.name, view


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="CMRxRecon 2023 undersampled root, e.g. .../TrainingSet/AccFactor04",
    )
    parser.add_argument("--acc-factor", choices=("04", "08", "10"), default="04")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Where to write per-slice phase .npy files.",
    )
    parser.add_argument(
        "--kc",
        type=int,
        default=8,
        help="Central k-space block side length (default 8 → 8×8 = 64 samples/frame).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if output .npy already exists.",
    )
    parser.add_argument(
        "--subject-filter",
        type=str,
        default=None,
        help="Optional substring to filter subject directories (e.g. 'P001').",
    )
    args = parser.parse_args()

    if not args.input_root.exists():
        print(f"ERROR: --input-root does not exist: {args.input_root}", file=sys.stderr)
        return 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "phase_manifest.csv"
    manifest_rows: list[dict] = []

    n_written = 0
    n_skipped = 0
    n_errors = 0

    for mat_path, subject, view in iter_cine_files(args.input_root):
        if args.subject_filter is not None and args.subject_filter not in subject:
            continue

        try:
            kspace = read_acquired_kspace(mat_path, args.acc_factor)
        except Exception as e:
            print(f"ERROR reading {mat_path}: {e}", file=sys.stderr)
            n_errors += 1
            manifest_rows.append(
                {
                    "subject": subject,
                    "view": view,
                    "slice": -1,
                    "T": 0,
                    "status": "error",
                    "message": str(e),
                    "output_path": "",
                }
            )
            continue

        if kspace is None:
            print(f"WARN: {mat_path} has no acquired k-space for AR {args.acc_factor}, skipping")
            manifest_rows.append(
                {
                    "subject": subject,
                    "view": view,
                    "slice": -1,
                    "T": 0,
                    "status": "missing_acquired_kspace",
                    "message": "",
                    "output_path": "",
                }
            )
            continue

        T, num_slices, C, H, W = kspace.shape
        subj_out_dir = args.output_root / args.acc_factor / subject
        subj_out_dir.mkdir(parents=True, exist_ok=True)

        for s in range(num_slices):
            out_path = subj_out_dir / f"{view}_slice{s}.npy"
            if out_path.exists() and not args.overwrite:
                n_skipped += 1
                manifest_rows.append(
                    {
                        "subject": subject,
                        "view": view,
                        "slice": s,
                        "T": T,
                        "status": "skip_existing",
                        "message": "",
                        "output_path": str(out_path),
                    }
                )
                continue

            try:
                phi = estimate_phase_from_kspace(kspace[:, s], kc=args.kc)
                np.save(out_path, phi)
                n_written += 1
                manifest_rows.append(
                    {
                        "subject": subject,
                        "view": view,
                        "slice": s,
                        "T": T,
                        "status": "written",
                        "message": "",
                        "output_path": str(out_path),
                    }
                )
                if n_written % 20 == 0:
                    print(f"  written {n_written}, skipped {n_skipped}, errors {n_errors}")
            except Exception as e:
                print(f"ERROR estimating {subject}/{view}/slice{s}: {e}", file=sys.stderr)
                n_errors += 1
                manifest_rows.append(
                    {
                        "subject": subject,
                        "view": view,
                        "slice": s,
                        "T": T,
                        "status": "error",
                        "message": str(e),
                        "output_path": "",
                    }
                )

    # Write manifest
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "view", "slice", "T", "status", "message", "output_path"],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print()
    print(f"Done. written={n_written}, skipped={n_skipped}, errors={n_errors}")
    print(f"Manifest: {manifest_path}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
