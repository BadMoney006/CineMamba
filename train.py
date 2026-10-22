import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import fastmri
from fastmri.losses import SSIMLoss
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import random
import numpy as np
from pathlib import Path

from cinemamba.data import CineCMRDataset
from cinemamba.model import CineMamba


# ================= Default configuration =================
# All values here can be overridden via CLI flags. See parse_args().
DEFAULT_CONFIG = {
    'root_path': None,
    'phase_root': None,
    'batch_size': 1,
    'num_epochs': 50,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'num_adj': 10,
    'acc_factor': 'all',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './checkpoints/baseline',
    'seed': 42,
    'frame_sampling': 'all',
    'num_sampled_frames': 3,
    'use_amp': False,
    'val_interval': 3,
    'val_num_subjects': 12,
    'val_frame_sampling': 'all',
    # Model config
    'num_cascades': 8,
    'n_feat0': 48,
    'img_size': 256,
    'use_checkpoint': True,
    # Innovation switches
    'use_t_to_batch': True,
    'use_cp_mamba': True,
    'use_mdsm': True,
    'ct_bidirectional': True,
    'ct_diastolic_anchor': True,
    'ct_num_phase_freqs': 4,
    'grad_clip': 0.1,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Train CineMamba on CMRxRecon 2023 Cine"
    )
    # Config identity
    p.add_argument("--run-name", type=str, required=True,
                   help="Experiment run name. Used as subdirectory of --save-root.")
    p.add_argument("--save-root", type=str, default="./runs",
                   help="Root for checkpoints, logs, metrics. Actual dir = save_root/run_name.")
    # Data
    p.add_argument("--root-path", nargs="+", required=True,
                   help="One or more CMRxRecon AccFactorNN directories.")
    p.add_argument("--phase-root", type=str, default=None,
                   help="ACS-derived phase arrays. Required for C3/C4.")
    p.add_argument("--acc-factor", type=str, default="all",
                   help="Acceleration factor to train on: '04', '08', '10', or 'all'.")
    # Windowing
    p.add_argument("--num-adj", type=int, default=DEFAULT_CONFIG['num_adj'],
                   help="Cardiac-cycle length T; the paper's 2023 experiments use T=10.")
    p.add_argument("--frame-sampling", type=str, default=DEFAULT_CONFIG['frame_sampling'],
                   choices=["all", "center", "random_one", "random_k"])
    # Training
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG['num_epochs'])
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG['batch_size'])
    p.add_argument("--lr", type=float, default=DEFAULT_CONFIG['lr'])
    p.add_argument("--min-lr", type=float, default=DEFAULT_CONFIG['min_lr'])
    p.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG['weight_decay'])
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG['seed'])
    p.add_argument("--val-interval", type=int, default=DEFAULT_CONFIG['val_interval'])
    p.add_argument("--val-num-subjects", type=int, default=DEFAULT_CONFIG['val_num_subjects'])
    p.add_argument("--grad-clip", type=float, default=DEFAULT_CONFIG['grad_clip'])
    # Model
    p.add_argument("--num-cascades", type=int, default=DEFAULT_CONFIG['num_cascades'])
    p.add_argument("--n-feat0", type=int, default=DEFAULT_CONFIG['n_feat0'])
    p.add_argument("--img-size", type=int, default=DEFAULT_CONFIG['img_size'])
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable gradient checkpointing (uses more VRAM but faster).")
    p.add_argument("--variant", choices=("C1", "C1b", "C2", "C3", "C4"), default="C4",
                   help="Paper ablation configuration; C4 is the complete model.")
    p.add_argument("--ct-no-bidirectional", action="store_true",
                   help="Directional ablation: use forward scan only.")
    p.add_argument("--ct-no-anchor", action="store_true",
                   help="Anchor ablation: initialize the scan at zero.")
    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from.")
    return p.parse_args()


def build_config(args):
    cfg = dict(DEFAULT_CONFIG)
    cfg['run_name'] = args.run_name
    cfg['variant'] = args.variant
    cfg['root_path'] = args.root_path
    cfg['phase_root'] = args.phase_root
    cfg['acc_factor'] = args.acc_factor
    cfg['num_adj'] = args.num_adj
    cfg['frame_sampling'] = args.frame_sampling
    cfg['num_epochs'] = args.epochs
    cfg['batch_size'] = args.batch_size
    cfg['lr'] = args.lr
    cfg['min_lr'] = args.min_lr
    cfg['weight_decay'] = args.weight_decay
    cfg['seed'] = args.seed
    cfg['val_interval'] = args.val_interval
    cfg['val_num_subjects'] = args.val_num_subjects
    cfg['grad_clip'] = args.grad_clip
    cfg['num_cascades'] = args.num_cascades
    cfg['n_feat0'] = args.n_feat0
    cfg['img_size'] = args.img_size
    cfg['use_checkpoint'] = not args.no_checkpoint
    cfg['use_t_to_batch'] = args.variant != 'C1'
    cfg['use_cp_mamba'] = args.variant in ('C2', 'C3', 'C4')
    cfg['use_mdsm'] = args.variant == 'C4'
    if args.num_adj != 10:
        raise ValueError("This CMRxRecon2023 release expects the paper's complete T=10 cycle")
    if args.variant in ('C3', 'C4') and not args.phase_root:
        raise ValueError(f"{args.variant} requires --phase-root with ACS-derived phase")
    cfg['ct_bidirectional'] = not args.ct_no_bidirectional
    cfg['ct_diastolic_anchor'] = not args.ct_no_anchor
    cfg['resume'] = args.resume
    cfg['save_dir'] = os.path.join(args.save_root, args.run_name)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_subjects(root_path):
    roots = root_path if isinstance(root_path, (list, tuple)) else [root_path]
    subjects = set()
    for root in roots:
        root_dir = Path(root)
        if not root_dir.exists():
            continue
        for d in root_dir.iterdir():
            if d.is_dir() and d.name.startswith('P'):
                subjects.add(d.name)
    return sorted(subjects)


def validate(model, loader, device, cfg):
    """Validation using the OFFICIAL PromptMR / CMRxRecon protocol:
    metrics computed on the whole (subject, view, slice, acc) VOLUME (all
    frames stacked), on the center-crop (H/3 x W/2) ROI, with data_range =
    global max of the cropped gt volume. Matches hellopipu/PromptMR
    promptmr_examples/cmrxrecon/utils.py, so numbers are directly comparable
    to PromptMR (~45 dB) and the CMRxRecon leaderboard.

    Assumes batch_size=1 (the training config). Collects the reconstructed
    center frame + its gt per sample, keyed by (subject, view, slice, acc);
    after the loop stacks frames into (T, H, W) volumes and computes metrics.

    Returns volume-averaged crop metrics: val_psnr/ssim/nmse (+ per-AR),
    val_n (num volumes). Keeps per-frame full-image PSNR/SSIM as
    val_psnr_full/val_ssim_full for continuity with earlier logs.
    """
    from collections import defaultdict
    from cinemamba.metrics import compute_cmrx_metrics

    model.eval()
    vols = defaultdict(dict)   # (subj, view, slice, acc) -> {frame_idx: (gt, pred)}
    frame_full = []            # (psnr_full, ssim_full, acc)
    with torch.no_grad():
        for batch in loader:
            input_k = batch['input_kspace'].to(device)
            target_k = batch['target_kspace'].to(device)
            mask = batch['mask'].to(device)

            if input_k.is_complex():
                input_k = torch.view_as_real(input_k)
            if target_k.is_complex():
                target_k = torch.view_as_real(target_k)

            acc_factor_idx = batch.get('acc_factor_idx', None)
            if acc_factor_idx is not None:
                acc_factor_idx = acc_factor_idx.to(device)
            phase = batch.get('phase', None)
            if phase is not None:
                phase = phase.to(device)

            IDX_TO_AR = {0: "04", 1: "08", 2: "10"}

            with autocast(enabled=False):
                if cfg['use_cp_mamba']:
                    output_seq = model(input_k, mask, acc_factor_idx=acc_factor_idx, phase=phase)
                else:
                    output_seq = model(input_k, mask, acc_factor_idx=acc_factor_idx)

            center_idx = cfg['num_adj'] // 2
            output_center = output_seq[:, center_idx, ...]                              # (B, H, W)
            target_img = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(target_k)), dim=1)  # (B, H, W)

            out_np = output_center.detach().cpu().numpy()
            gt_np = target_img.detach().cpu().numpy()

            meta = batch.get('metadata', {})
            for i in range(out_np.shape[0]):
                def _mv(field, default=None):
                    v = meta.get(field, default)
                    if isinstance(v, (list, tuple)):
                        return v[i]
                    if torch.is_tensor(v):
                        return v[i].item()
                    return v
                subj = _mv('subject', 'unk')
                view = _mv('view', 'unk')
                sl = _mv('slice', 0)
                fidx = _mv('frame_idx', 0)
                acc = _mv('acc_factor', None)
                if acc is not None:
                    acc = str(acc).zfill(2)
                elif acc_factor_idx is not None:
                    acc = IDX_TO_AR.get(int(acc_factor_idx[i].item()), "unknown")
                else:
                    acc = "unknown"

                key = (subj, view, int(sl), acc)
                vols[key][int(fidx)] = (gt_np[i], out_np[i])

                mv = float(gt_np[i].max()) or 1.0
                frame_full.append((
                    psnr_metric(gt_np[i], out_np[i], data_range=mv),
                    ssim_metric(gt_np[i], out_np[i], data_range=mv),
                    acc,
                ))

    model.train()

    if not vols:
        return {}

    vol_records = []  # (psnr, ssim, nmse, acc)
    for key, frames in vols.items():
        acc = key[3]
        fidxs = sorted(frames.keys())
        gt_vol = np.stack([frames[f][0] for f in fidxs], axis=0)
        pred_vol = np.stack([frames[f][1] for f in fidxs], axis=0)
        m = compute_cmrx_metrics(gt_vol, pred_vol, do_crop=True)
        vol_records.append((m['psnr'], m['ssim'], m['nmse'], acc))

    def _agg(recs, idx):
        return float(np.mean([r[idx] for r in recs])) if recs else None

    result = {
        'val_psnr': _agg(vol_records, 0),      # crop, whole-volume — PromptMR-comparable
        'val_ssim': _agg(vol_records, 1),
        'val_nmse': _agg(vol_records, 2),
        'val_n': len(vol_records),
        'val_psnr_full': _agg(frame_full, 0),  # per-frame full-image (continuity)
        'val_ssim_full': _agg(frame_full, 1),
    }
    for ar in ("04", "08", "10"):
        sub = [r for r in vol_records if r[3] == ar]
        if sub:
            result[f'val_psnr_ar{ar}'] = _agg(sub, 0)
            result[f'val_ssim_ar{ar}'] = _agg(sub, 1)
            result[f'val_nmse_ar{ar}'] = _agg(sub, 2)
            result[f'val_n_ar{ar}'] = len(sub)
    return result


def log_line(log_path, obj):
    """Append one JSON object per line to the JSONL log."""
    with open(log_path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def train(cfg):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    set_seed(cfg['seed'])
    device = torch.device(cfg['device'])
    print(f"🚀 Device: {device}  |  run_name={cfg['run_name']}")

    # Persist the resolved config for reproducibility
    with open(os.path.join(cfg['save_dir'], "config.json"), "w") as f:
        json.dump({k: v for k, v in cfg.items() if k != 'device'}, f, indent=2)

    log_path = os.path.join(cfg['save_dir'], "log.jsonl")
    # Truncate log if not resuming
    if not cfg.get('resume'):
        open(log_path, "w").close()

    # Data
    print("📂 Loading dataset...")
    subjects = list_subjects(cfg['root_path'])
    val_num = min(cfg['val_num_subjects'], len(subjects))
    rng = random.Random(cfg['seed'])
    rng.shuffle(subjects)
    if val_num <= 0 or val_num >= len(subjects):
        val_subjects = set()
        train_subjects = set(subjects)
    else:
        val_subjects = set(subjects[:val_num])
        train_subjects = set(subjects[val_num:])

    train_dataset = CineCMRDataset(
        root_dir=cfg['root_path'],
        acc_factor=cfg['acc_factor'],
        num_adj=cfg['num_adj'],
        is_train=True,
        frame_sampling=cfg['frame_sampling'],
        num_sampled_frames=cfg['num_sampled_frames'],
        seed=cfg['seed'],
        include_subjects=train_subjects if train_subjects else None,
        phase_root=cfg['phase_root'],
        require_phase=cfg['variant'] in ('C3', 'C4'),
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    print(f"✅ Train samples: {len(train_dataset)}")

    val_loader = None
    if val_subjects:
        print(f"✅ Val subjects: {len(val_subjects)}")
        val_dataset = CineCMRDataset(
            root_dir=cfg['root_path'],
            acc_factor=cfg['acc_factor'],
            num_adj=cfg['num_adj'],
            is_train=False,
            frame_sampling=cfg['val_frame_sampling'],
            num_sampled_frames=cfg['num_sampled_frames'],
            seed=cfg['seed'],
            include_subjects=val_subjects,
            phase_root=cfg['phase_root'],
            require_phase=cfg['variant'] in ('C3', 'C4'),
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
        print(f"✅ Val samples: {len(val_dataset)}")

    # Model
    print("Initializing CineMamba...")
    ct_kwargs = dict(
        bidirectional=cfg['ct_bidirectional'],
        diastolic_anchor=cfg['ct_diastolic_anchor'],
        num_phase_freqs=cfg['ct_num_phase_freqs'],
    )
    model = CineMamba(
        num_cascades=cfg['num_cascades'],
        num_adj_slices=cfg['num_adj'],
        n_feat0=cfg['n_feat0'],
        img_size=cfg['img_size'],
        use_checkpoint=cfg['use_checkpoint'],
        use_t_to_batch=cfg['use_t_to_batch'],
        use_cp_mamba=cfg['use_cp_mamba'],
        cp_mamba_kwargs=ct_kwargs,
        use_mdsm=cfg['use_mdsm'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Trainable params: {n_params / 1e6:.2f} M")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['num_epochs'], eta_min=cfg['min_lr']
    )
    criterion_ssim = SSIMLoss().to(device)

    start_epoch = 0
    if cfg.get('resume'):
        ckpt = torch.load(cfg['resume'], map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', 0)
        else:
            model.load_state_dict(ckpt)
        print(f"🔁 Resumed from {cfg['resume']} at epoch {start_epoch}")

    print("🔥 Training start!")
    best_psnr = -float('inf')
    for epoch in range(start_epoch, cfg['num_epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")

        for i, batch in enumerate(progress_bar):
            input_k = batch['input_kspace'].to(device)
            target_k = batch['target_kspace'].to(device)
            mask = batch['mask'].to(device)

            if input_k.is_complex():
                input_k = torch.view_as_real(input_k)
            if target_k.is_complex():
                target_k = torch.view_as_real(target_k)

            acc_factor_idx = batch.get('acc_factor_idx', None)
            if acc_factor_idx is not None:
                acc_factor_idx = acc_factor_idx.to(device)
            phase = batch.get('phase', None)
            if phase is not None:
                phase = phase.to(device)

            with autocast(enabled=False):
                if cfg['use_cp_mamba']:
                    output_seq = model(input_k, mask, acc_factor_idx=acc_factor_idx, phase=phase)
                else:
                    output_seq = model(input_k, mask, acc_factor_idx=acc_factor_idx)

            center_idx = cfg['num_adj'] // 2
            output_center = output_seq[:, center_idx, ...]

            target_img = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(target_k)), dim=1)

            with autocast(enabled=False):
                output_center_f = output_center.float()
                target_img_f = target_img.float()
                # Pure SSIM loss (matches PromptMR's fastmri.SSIMLoss). The prior
                # L1+SSIM combo was effectively SSIM-only anyway: MRI RSS pixel
                # magnitudes are ~1e-4, so the L1 term was ~5 orders below SSIM
                # and contributed no gradient. data_range = per-sample gt max,
                # consistent with the eval protocol (cmrx_eval.psnr maxval=gt.max()).
                max_val = target_img_f.reshape(target_img_f.shape[0], -1).max(dim=1)[0].clamp(min=1e-8)
                loss_ssim = criterion_ssim(
                    output_center_f.unsqueeze(1), target_img_f.unsqueeze(1), data_range=max_val
                )
                loss = loss_ssim

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['grad_clip'])
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ssim': f"{loss_ssim.item():.4f}",
            })

        avg_loss = epoch_loss / max(1, len(train_loader))
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.6f}  time={epoch_time:.1f}s")

        record = {
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'time_s': epoch_time,
            'lr': scheduler.get_last_lr()[0],
        }

        if val_loader and (epoch + 1) % cfg['val_interval'] == 0:
            val_metrics = validate(model, val_loader, device, cfg)
            if val_metrics:
                # Overall — show both full-image and CMRxRecon center-crop ROI.
                vp = val_metrics.get('val_psnr')        # crop, whole-volume (PromptMR口径)
                vs = val_metrics.get('val_ssim')
                vnm = val_metrics.get('val_nmse')
                vpf = val_metrics.get('val_psnr_full')  # per-frame full-image (continuity)
                vsf = val_metrics.get('val_ssim_full')
                print(f"  [CMRx crop/vol] psnr={vp:.4f} ssim={vs:.4f} nmse={vnm:.5f}   "
                      f"[full/frame] psnr={vpf:.4f} ssim={vsf:.4f}   (n_vol={val_metrics.get('val_n')})")
                # Per-AR (all CMRxRecon crop/volume protocol — leaderboard-comparable)
                for ar in ("04", "08", "10"):
                    pk, sk, nk = f'val_psnr_ar{ar}', f'val_ssim_ar{ar}', f'val_n_ar{ar}'
                    if pk in val_metrics:
                        print(f"    AR{ar}: psnr={val_metrics[pk]:.4f}  ssim={val_metrics[sk]:.4f}  n_vol={val_metrics[nk]}")
                record.update(val_metrics)
                # Save best by CMRxRecon crop/volume PSNR (leaderboard metric).
                best_metric = val_metrics.get('val_psnr', None)
                if best_metric is not None and best_metric > best_psnr:
                    best_psnr = best_metric
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch + 1,
                        'val_metrics': val_metrics,
                    }, os.path.join(cfg['save_dir'], 'best_psnr.pth'))

        scheduler.step()
        log_line(log_path, record)

        # Save latest every epoch (small file, fast recovery)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
        }, os.path.join(cfg['save_dir'], 'latest.pth'))

        # Periodic milestone
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch + 1,
            }, os.path.join(cfg['save_dir'], f'epoch_{epoch+1}.pth'))

    print("✅ Training finished.")


if __name__ == "__main__":
    args = parse_args()
    cfg = build_config(args)
    train(cfg)
