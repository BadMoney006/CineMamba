import h5py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn.functional as F

ACC_FACTOR_MAP = {'04': 0, '08': 1, '10': 2}


class CineCMRDataset(Dataset):
    def __init__(
        self,
        root_dir,
        acc_factor='04',
        num_adj=10,
        is_train=True,
        transform=None,
        frame_sampling="all",
        num_sampled_frames=3,
        seed=42,
        include_subjects=None,
        exclude_subjects=None,
        phase_root=None,
        require_phase=False,
        require_full_cycle=True,
    ):
        """
        CMRxRecon2023 Cine loader for complete ten-frame cycles.

        Args:
            phase_root: Optional directory containing per-slice cardiac phase
                arrays produced by ``python -m cinemamba.phase``. Layout:
                    <phase_root>/<acc_factor>/<subject>/<view>_slice<slice_idx>.npy
                Each .npy has shape (T,) with phi_t in [0, 1). If provided,
                __getitem__ returns a 'phase' key of shape (num_adj,) aligned
                with the adjacent-frame window. C2 uses uniform t/T;
                C3/C4 require ACS-derived arrays.
        """
        self.root_dirs = self._normalize_root_dirs(root_dir)
        self.acc_factor = acc_factor
        self.acc_factors = self._normalize_acc_factors(acc_factor)
        self.num_adj = num_adj
        self.is_train = is_train
        self.transform = transform
        self.frame_sampling = frame_sampling
        self.num_sampled_frames = num_sampled_frames
        self.rng = np.random.RandomState(seed)
        self.include_subjects = set(include_subjects) if include_subjects is not None else None
        self.exclude_subjects = set(exclude_subjects) if exclude_subjects is not None else None
        self.phase_root = Path(phase_root) if phase_root is not None else None
        self.require_phase = require_phase
        self.require_full_cycle = require_full_cycle
        self._phase_cache = {}
        self._warned_missing_phase = set()
        
        # 自动推断 FullSample 目录 (per root)
        self.full_dirs = {}
        for root in self.root_dirs:
            full_dir = root.parent / 'FullSample'
            if not full_dir.exists():
                raise FileNotFoundError(f"Reference FullSample directory is required for supervised evaluation: {full_dir}")
            self.full_dirs[root] = full_dir

        self.samples = []
        self._mask_cache = {}
        self._warned_missing_mask = set()
        self._scan_dataset()

    def _normalize_root_dirs(self, root_dir):
        if isinstance(root_dir, (list, tuple)):
            return [Path(p) for p in root_dir]
        return [Path(root_dir)]

    def _normalize_acc_factors(self, acc_factor):
        if acc_factor is None:
            return None
        if isinstance(acc_factor, str):
            if acc_factor.lower() in ("all", "*"):
                return None
            return [acc_factor]
        if isinstance(acc_factor, (list, tuple, set)):
            return [str(a) for a in acc_factor]
        return [str(acc_factor)]

    def _available_acc_factors(self, keys):
        accs = set()
        for k in keys:
            if k.startswith("kspace_single_sub"):
                acc = k[len("kspace_single_sub"):]
                if acc:
                    accs.add(acc)
            elif k.startswith("kspace_sub"):
                acc = k[len("kspace_sub"):]
                if acc:
                    accs.add(acc)
        return accs

    def _scan_dataset(self):
        for root_dir in self.root_dirs:
            print(f"正在扫描数据集: {root_dir} ...")
            subject_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith('P')])
            full_dir = self.full_dirs.get(root_dir)
            
            for subj_dir in subject_dirs:
                subj_name = subj_dir.name
                if self.include_subjects is not None and subj_name not in self.include_subjects:
                    continue
                if self.exclude_subjects is not None and subj_name in self.exclude_subjects:
                    continue
                for view in ['sax', 'lax']:
                    fname = f"cine_{view}.mat"
                    file_path = subj_dir / fname
                    
                    full_path = None
                    if full_dir:
                        full_path = full_dir / subj_dir.name / fname
                        if not full_path.exists():
                            continue

                    if file_path.exists():
                        try:
                            with h5py.File(file_path, 'r') as f:
                                keys = list(f.keys())
                                available_accs = self._available_acc_factors(keys)
                                if self.acc_factors is None:
                                    acc_list = sorted(available_accs)
                                else:
                                    acc_list = [a for a in self.acc_factors if a in available_accs]
                                
                                for acc in acc_list:
                                    target_key = None
                                    if f'kspace_sub{acc}' in keys:
                                        target_key = f'kspace_sub{acc}'
                                    elif f'kspace_single_sub{acc}' in keys:
                                        target_key = f'kspace_single_sub{acc}'
                                    
                                    if target_key:
                                        # --- 修复核心：兼容 Group 和 Dataset ---
                                        obj = f[target_key]
                                        shape = None
                                        try:
                                            shape = obj['real'].shape # 尝试读取结构体
                                        except:
                                            shape = obj.shape # 直接读取
                                        # -----------------------------------
                                        
                                        num_t = shape[0]
                                        num_slices = shape[1]
                                        frame_indices = self._get_frame_indices(num_t)
                                        for s in range(num_slices):
                                            for t in frame_indices:
                                                self.samples.append({
                                                    'sub_path': str(file_path),
                                                    'full_path': str(full_path),
                                                    'slice_idx': s,
                                                    'frame_idx': t,
                                                    'num_frames': num_t,
                                                    'subject': subj_name,
                                                    'view': view,
                                                    'acc_factor': acc,
                                                })
                        except Exception as e:
                            print(f"读取错误 {file_path}: {e}")
        print(f"数据集扫描完成: 共找到 {len(self.samples)} 个样本。")

    def _read_kspace(self, file_path, key_suffix, slice_idx):
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            is_single = 'kspace_single_full' in keys or f'kspace_single_{key_suffix}' in keys
            prefix = 'kspace_single_' if is_single else 'kspace_'
            target_key = f"{prefix}{key_suffix}"
            
            data_ptr = f[target_key]
            try:
                real = data_ptr['real'][:]
                imag = data_ptr['imag'][:]
                kspace = real + 1j * imag
            except:
                kspace = data_ptr[:]

            kspace = kspace[:, slice_idx, ...]
            if kspace.ndim == 4:
                kspace = kspace.transpose(0, 1, 3, 2)
            elif kspace.ndim == 3:
                kspace = kspace.transpose(0, 2, 1)
            return kspace

    def _get_frame_indices(self, num_t):
        mode = (self.frame_sampling or "all").lower()
        if mode == "all":
            return list(range(num_t))
        if mode == "center":
            return [num_t // 2]
        if mode == "random_one":
            return [-1]
        if mode == "random_k":
            k = max(1, int(self.num_sampled_frames))
            k = min(k, num_t)
            return sorted(self.rng.choice(num_t, size=k, replace=False).tolist())
        return list(range(num_t))

    def _read_mask(self, file_path):
        if file_path in self._mask_cache:
            return self._mask_cache[file_path]
        mask_path = file_path.replace(".mat", "_mask.mat")
        if not os.path.exists(mask_path):
            if file_path not in self._warned_missing_mask:
                print(f"⚠️ 未找到 mask 文件，回退使用非零 mask: {mask_path}")
                self._warned_missing_mask.add(file_path)
            self._mask_cache[file_path] = None
            return None
        with h5py.File(mask_path, "r") as f:
            key = list(f.keys())[0]
            mask = f[key][()]
        if mask.ndim >= 2:
            mask = mask[:, 0]
        mask = np.asarray(mask).squeeze()
        self._mask_cache[file_path] = mask
        return mask

    def _load_phase_full_slice(self, subject, view, slice_idx, num_frames, acc_factor):
        """Load precomputed phi ∈ [0, 1)^T for one (subject, view, slice).

        Returns:
            np.ndarray of shape (T,) float32. If phase_root is None or the
            file is missing, returns the uniform fallback t/T so that the
            downstream Δ generator still has a well-defined signal.
        """
        cache_key = (acc_factor, subject, view, slice_idx)
        if cache_key in self._phase_cache:
            return self._phase_cache[cache_key]

        if self.phase_root is None:
            if self.require_phase:
                raise ValueError("C3/C4 require --phase-root with ACS-derived phase arrays")
            phi = (np.arange(num_frames, dtype=np.float32) / max(num_frames, 1)) % 1.0
            self._phase_cache[cache_key] = phi
            return phi

        phase_path = self.phase_root / acc_factor / subject / f"{view}_slice{slice_idx}.npy"
        if not phase_path.exists():
            if self.require_phase:
                raise FileNotFoundError(f"Missing ACS-derived cardiac phase: {phase_path}")
            if str(phase_path) not in self._warned_missing_phase:
                print(
                    f"⚠️ 未找到 phase 文件, 回退到 t/T: {phase_path}"
                )
                self._warned_missing_phase.add(str(phase_path))
            phi = (np.arange(num_frames, dtype=np.float32) / max(num_frames, 1)) % 1.0
            self._phase_cache[cache_key] = phi
            return phi

        phi = np.load(phase_path).astype(np.float32)
        # Never silently alter the phase vector for the paper configuration.
        if phi.shape[0] != num_frames:
            if self.require_phase:
                raise ValueError(f"Phase length {phi.shape[0]} != image frames {num_frames}: {phase_path}")
            if phi.shape[0] > num_frames:
                phi = phi[:num_frames]
            else:
                # Pad by cyclic repetition — cardiac phase is naturally periodic.
                reps = int(np.ceil(num_frames / phi.shape[0]))
                phi = np.tile(phi, reps)[:num_frames]
        self._phase_cache[cache_key] = phi.astype(np.float32)
        return self._phase_cache[cache_key]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        acc = sample.get('acc_factor', self.acc_factor)
        kspace_sub = self._read_kspace(sample['sub_path'], f"sub{acc}", sample['slice_idx'])
        kspace_full = self._read_kspace(sample['full_path'], "full", sample['slice_idx'])
        
        num_frames = kspace_sub.shape[0]
        if self.require_full_cycle and num_frames != self.num_adj:
            raise ValueError(
                f"Expected one complete {self.num_adj}-frame cycle, got {num_frames}"
            )
        t = sample.get('frame_idx', num_frames // 2)
        if t is None or int(t) < 0:
            t = int(self.rng.randint(0, num_frames))

        # Build a length-num_adj adjacent-frame window centered on `t`.
        # For odd num_adj (e.g. 5, 7), symmetric: t + [-2, -1, 0, 1, 2].
        # For even num_adj (e.g. 8), take one extra frame on the LEFT so
        # the returned window is exactly num_adj frames:
        #   num_adj=8, center_idx=4  → offsets [-4,-3,-2,-1,0,1,2,3]  (t is at index 4)
        if self.num_adj % 2 == 1:
            center_idx = self.num_adj // 2
            offsets = range(-center_idx, center_idx + 1)          # length = num_adj
        else:
            center_idx = self.num_adj // 2                        # e.g. 4 for num_adj=8
            offsets = range(-center_idx, center_idx)              # length = num_adj (asymmetric)
        adj_indices = [ (t + offset) % num_frames for offset in offsets ]
        assert len(adj_indices) == self.num_adj, \
            f"adj_indices length {len(adj_indices)} != num_adj {self.num_adj}"

        kspace_sub_adj = kspace_sub[adj_indices]
        kspace_full_center = kspace_full[t]

        # Load cardiac phase for this (subject, view, slice) and gather the
        # window aligned with adj_indices.
        phi_full = self._load_phase_full_slice(
            subject=sample['subject'],
            view=sample['view'],
            slice_idx=sample['slice_idx'],
            num_frames=num_frames,
            acc_factor=acc,
        )
        phase_adj = phi_full[np.asarray(adj_indices)].astype(np.float32)
        
        kspace_sub_tensor = torch.from_numpy(kspace_sub_adj).cfloat()
        kspace_full_tensor = torch.from_numpy(kspace_full_center).cfloat()

        # Ensure coil dimension exists before building mask
        if kspace_sub_tensor.dim() == 3:
            kspace_sub_tensor = kspace_sub_tensor.unsqueeze(1)
            kspace_full_tensor = kspace_full_tensor.unsqueeze(0)
        
        # mask = torch.abs(kspace_sub_tensor).sum(dim=-1).sum(dim=-1) > 0
        # [修改后] 这是空间 Mask，正确反映采样位置
        # 只要该像素点幅值大于0，就认为是被采样的
        mask_1d = self._read_mask(sample['sub_path'])
        if mask_1d is not None:
            num_cols = kspace_sub_tensor.shape[-2]
            mask_1d = mask_1d.astype(np.float32)
            if mask_1d.shape[0] != num_cols:
                raise ValueError(f"Mask length {mask_1d.shape[0]} != k-space width {num_cols}")
            mask_col = torch.from_numpy(mask_1d).view(1, 1, num_cols, 1)
            mask = mask_col.expand(kspace_sub_tensor.shape[0], kspace_sub_tensor.shape[1], num_cols, kspace_sub_tensor.shape[-1])
        else:
            mask = (torch.abs(kspace_sub_tensor) > 0).float()
        mask = mask.float()
        
        # 单线圈增加维度: (T, H, W) -> (T, 1, H, W)
        # --- 新增：Padding 到 16 的倍数 (防止 U-Net 尺寸不匹配报错) ---
        # 目标: H=246 -> 256
        pad_h = (16 - kspace_sub_tensor.shape[-2] % 16) % 16
        pad_w = (16 - kspace_sub_tensor.shape[-1] % 16) % 16
        
        if pad_h > 0 or pad_w > 0:
            # F.pad 参数顺序: (Left, Right, Top, Bottom)
            # 记得对 kspace_sub, kspace_full, mask 都做 padding
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            kspace_sub_tensor = F.pad(kspace_sub_tensor, padding, mode='constant', value=0)
            kspace_full_tensor = F.pad(kspace_full_tensor, padding, mode='constant', value=0)
            mask = F.pad(mask, padding, mode='constant', value=0)
        # -----------------------------------------------------------

        acc_str = sample.get('acc_factor', '04')
        acc_idx = ACC_FACTOR_MAP.get(str(acc_str).zfill(2), 0)

        return {
            'input_kspace': kspace_sub_tensor, # (T, Coil, H, W)
            'target_kspace': kspace_full_tensor, # (Coil, H, W)
            'mask': mask,
            'acc_factor_idx': torch.tensor(acc_idx, dtype=torch.long),
            'phase': torch.from_numpy(phase_adj),  # (num_adj,) float32, cardiac phase in [0, 1)
            'metadata': {
                'subject': sample['subject'],
                'view': sample['view'],
                'slice': sample['slice_idx'],
                'frame_idx': t,
                'acc_factor': acc,
            }
        }
