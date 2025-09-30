from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
import ast
from glob import glob
import os
from pathlib import Path
import argparse

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def display_dataset_info(datadir, dataset):      
    print(f'Dataset path: {datadir}')    
    if dataset is not None:
        print(f"Found {len(dataset)} images.")    

def load_state_dict(model, state_dict):
    """
    model.module vs model key mismatch 문제를 자동으로 해결
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()

    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)

    for k, v in state_dict.items():
        if is_ddp:
            if not k.startswith('module.'):
                k = 'module.' + k
        else:
            if k.startswith('module.'):
                k = k[len('module.'):]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(new_state_dict.keys()) & model_keys

    total = len(model_keys)
    loaded = len(loaded_keys)
    percent = 100.0 * loaded / total if total > 0 else 0.0

    print(f"[Info] Loaded {loaded}/{total} state_dict entries ({percent:.2f}%) from checkpoint.")

class SegmentationTransform:
    def __init__(
        self,
        crop_size=(1024, 1024),
        scale_range=(0.5, 1.5),
        is_train=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        prob_by_tag=None,   # {'low_light':0.0, 'overbright':0.0, 'degradation':0.0, 'normal':0.0}
        fn_by_tag=None,      # {'low_light': callable(img,label)->(img,label), ...}
        val_resize_size=(1080, 1920)
    ):
        self.crop_size = crop_size  # (H, W)
        self.scale_range = scale_range
        self.is_train = is_train
        self.val_resize_size = val_resize_size

        self.mean = mean
        self.std = std

        self.bilinear = transforms.InterpolationMode.BILINEAR
        self.nearest  = transforms.InterpolationMode.NEAREST

        # 태그별 확률/함수 (기본 no-op)
        base_prob = {"low_light":0.0, "overbright":0.0, "degradation":0.0, "normal":0.0}
        self.prob = base_prob if prob_by_tag is None else {**base_prob, **prob_by_tag}
        self.fn = {k: None for k in base_prob}
        if fn_by_tag:
            for k, f in fn_by_tag.items():
                if k in self.fn:
                    self.fn[k] = f

    # ---- 내부 유틸: 기하 증강 ----
    def _random_scale(self, image, label):
        s = random.uniform(self.scale_range[0], self.scale_range[1])
        w, h = image.size
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        image = TF.resize(image, (nh, nw), interpolation=self.bilinear)
        label = TF.resize(label, (nh, nw), interpolation=self.nearest)
        return image, label

    def _pad_and_random_crop(self, image, label):
        H, W = self.crop_size
        _, h, w = 0, image.size[1], image.size[0]
        pad_h, pad_w = max(H - h, 0), max(W - w, 0)
        if pad_h > 0 or pad_w > 0:
            # left, top, right, bottom
            image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
            label = TF.pad(label, (0, 0, pad_w, pad_h), fill=255)
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)
        return image, label

    def _random_hflip(self, image, label, p=0.5):
        if random.random() < p:
            image = TF.hflip(image)
            label = TF.hflip(label)
        return image, label

    def _resize_to(self, image, label, size_hw):
        H, W = int(size_hw[0]), int(size_hw[1])
        image = TF.resize(image, (H, W), interpolation=self.bilinear)
        label = TF.resize(label, (H, W), interpolation=self.nearest)
        return image, label

    # ---- 태그 훅: 확률만 관리(기본 no-op) ----
    def _maybe_apply_tag_aug(self, image, label, tag):
        p = self.prob.get(tag, 0.0)
        if random.random() < p:
            fn = self.fn.get(tag, None)
            if fn is not None:
                return fn(image, label)  # 사용자가 나중에 채울 자리
        return image, label

    def __call__(self, image, label, tag="normal"):
        if not self.is_train:
            # VALID: ToTensor + Normalize 만
            if self.val_resize_size is not None:
                image, label = self._resize_to(image, label, self.val_resize_size)

            img_t = TF.to_tensor(image)
            img_t = TF.normalize(img_t, self.mean, self.std)
            lab_t = torch.from_numpy(np.array(label, dtype=np.uint8)).long()
            return img_t, lab_t

        # TRAIN: 기하 증강
        image, label = self._random_scale(image, label)
        image, label = self._pad_and_random_crop(image, label)
        image, label = self._random_hflip(image, label, p=0.5)

        # TRAIN: 태그별 증강 (자리만 남김)
        image, label = self._maybe_apply_tag_aug(image, label, tag)

        # ToTensor + Normalize
        img_t = TF.to_tensor(image)
        img_t = TF.normalize(img_t, self.mean, self.std)
        lab_t = torch.from_numpy(np.array(label, dtype=np.uint8)).long()
        return img_t, lab_t


# =========================
# 통합 Dataset (train/val 공용)
# - subset: 'train' | 'val'
# - 라벨 경로: image → labelmap 치환, 파일명 동일(.png 가정)
# - 태그: 경로에서 폴더명 탐색 (low_light / overbright / degradation / normal)
# =========================
class SegmentationDataset(Dataset):
    TAG_ALIASES = {
        "lowlight": "low_light",
        "low_light": "low_light",
        "overlight": "overbright",
        "overbright": "overbright",
        "degradation": "degradation",
        "normal": "normal",
    }
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root_dir,
        crop_size=(1024,1024),
        subset="train",                 # 'train' or 'val'
        scale_range=(0.5, 1.5),
        prob_by_tag=None,
        fn_by_tag=None,
        val_resize_size=(1080, 1920)
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.subset = subset

        # 이미지/라벨 경로 수집
        img_pattern = os.path.join(self.root_dir, subset, "*", "image", "**", "*.*")
        all_imgs = sorted(glob(img_pattern, recursive=True))

        image_paths, label_paths, tags = [], [], []
        for p in all_imgs:
            suffix = Path(p).suffix.lower()
            if suffix not in self.IMG_EXTS:
                continue
            lp = self._get_label_path(p)  # same tag, labelmap로 치환
            if not os.path.exists(lp):
                # 필요하면 경고만 출력하고 continue
                # print(f"[WARN] label not found for {p}")
                continue
            image_paths.append(p)
            label_paths.append(lp)
            tags.append(self._get_tag(p))

        self.image_paths = image_paths
        self.label_paths = label_paths
        self.tags = tags

        # Transform: subset에 따라 train/val 전환
        self.transform = SegmentationTransform(
            crop_size=crop_size,
            scale_range=scale_range,
            is_train=(subset == "train"),
            prob_by_tag=prob_by_tag,
            fn_by_tag=fn_by_tag,
            val_resize_size=val_resize_size
        )

    # --- image → labelmap 치환 (같은 subset/tag 하위로)
    def _get_label_path(self, image_path: str) -> str:
        """
        .../<subset>/<tag>/image/aaa/bbb/CCC.ext
        →   .../<subset>/<tag>/labelmap/aaa/bbb/CCC.png
        """
        p = Path(image_path)
        parts = list(p.parts)
        # subset 인덱스 찾기
        sub_idx = parts.index(self.subset)
        tag = parts[sub_idx + 1]  # low_light, overbright, ...
        # image 다음의 상대경로 (파일명 포함)
        assert parts[sub_idx + 2].lower() == "image"
        rel_inside = Path(*parts[(sub_idx + 3):])  # aaa/bbb/CCC.ext
        # 라벨 경로 조립 (확장자는 .png로 통일)
        lbl = Path(self.root_dir, self.subset, tag, "labelmap", rel_inside).with_suffix(".png")
        return str(lbl)

    # --- 경로에서 태그 추출
    def _get_tag(self, image_path: str) -> str:
        parts = [s.lower() for s in Path(image_path).parts]
        sub_idx = parts.index(self.subset)
        raw = parts[sub_idx + 1]
        return self.TAG_ALIASES.get(raw, "normal")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        lab = Image.open(self.label_paths[idx]).convert("L")
        tag = self.tags[idx]
        img_t, lab_t = self.transform(img, lab, tag=tag)
        return img_t, lab_t



class CrossEntropy(nn.Module):
    def __init__(self, ignore_label= 255, weight= None, aux_weights = [1, 0.4]):
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds, labels):
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels):
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)    
    
    
class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label= 255, weight = None, thresh = 0.6, aux_weights= [1, 0.4]):
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds, labels):
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels):
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)        

import torch.nn as nn
import torch.nn.functional as F

# 🔹 새로 추가
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=255, reduction='mean', aux_weights=[1.0, 0.4]):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.aux_weights = aux_weights

    def _forward_single(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        logpt = -ce_loss
        pt = torch.exp(logpt)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def forward(self, logits, targets):
        if isinstance(logits, tuple):  # main_out, aux_out 같이 들어오는 경우
            return sum(w * self._forward_single(l, targets) for l, w in zip(logits, self.aux_weights))
        else:
            return self._forward_single(logits, targets)


# focal loss에서 필요한 연산

import torch
import numpy as np
from tqdm import tqdm


def compute_class_weights(dataloader, num_classes, ignore_index=255, method="inverse"):
    """
    dataloader: 학습용 DataLoader (train_loader)
    num_classes: 클래스 개수 (예: 19)
    ignore_index: 무시할 라벨 (보통 255)
    method: "inverse" | "effective_num"

    반환: torch.Tensor [num_classes]
    """
    counts = np.zeros(num_classes, dtype=np.int64)

    for imgs, labels, _ in tqdm(dataloader, desc="Counting class frequencies"):
        labels = labels.numpy()
        for c in range(num_classes):
            counts[c] += np.sum(labels == c)

    # ----- 방법 1: 단순 역비율 (Inverse Frequency) -----
    if method == "inverse":
        weights = 1.0 / (counts + 1e-6)  # 빈도가 적을수록 큰 가중치
        weights = weights / weights.sum() * num_classes  # 정규화

    # ----- 방법 2: Effective Number of Samples (Cui et al., CVPR 2019) -----
    elif method == "effective_num":
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-6)
        weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)

import torch
import os

def compute_or_load_class_weights(dataloader, num_classes, cache_path=None,
                                  ignore_index=255, method="effective_num"):
    # 캐시 파일이 있으면 바로 불러오기
    if os.path.exists(cache_path):
        print(f"[Info] Loading precomputed class weights from {cache_path}")
        return torch.load(cache_path)

    # 없으면 새로 계산
    print("[Info] Computing class weights...")
    weights = compute_class_weights(dataloader, num_classes,
                                    ignore_index=ignore_index, method=method)
    torch.save(weights, cache_path)
    print(f"[Info] Saved class weights to {cache_path}")
    return weights

from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs=10, eta_min=0, last_epoch=-1):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # linear warmup: from 0 to base_lr
            return [
                base_lr * float(self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs
            ]    
    


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, total_epochs=500, decay_epoch=1, power=0.9, last_epoch=-1) -> None:
        self.decay_epoch = decay_epoch
        self.total_epochs = total_epochs
        self.power = power
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_epoch != 0 or self.last_epoch > self.total_epochs:
            return self.base_lrs
        else:
            factor = (1 - self.last_epoch / float(self.total_epochs)) ** self.power
            return [factor*lr for lr in self.base_lrs]

class EpochWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs=5, warmup_ratio=5e-4, warmup='linear',total_epochs=500, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        return [max(ratio * lr, 1e-7) for lr in self.base_lrs]

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_epochs:
            return self.get_warmup_ratio()
        return self.get_main_ratio()

    def get_warmup_ratio(self):
        alpha = self.last_epoch / self.warmup_epochs
        if self.warmup == 'linear':
            return self.warmup_ratio + (1. - self.warmup_ratio) * alpha
        else:
            return self.warmup_ratio ** (1. - alpha)

    def get_main_ratio(self):
        raise NotImplementedError
        
        
class WarmupPolyEpochLR(EpochWarmupLR):
    def __init__(self, optimizer, power=0.9, total_epochs=500, warmup_epochs=5, warmup_ratio=5e-4, warmup='linear', last_epoch=-1):
        self.power = power
        super().__init__(optimizer, warmup_epochs, warmup_ratio, warmup, total_epochs, last_epoch)

    def get_main_ratio(self):
        real_epoch = self.last_epoch - self.warmup_epochs
        real_total = self.total_epochs - self.warmup_epochs
        alpha = min(real_epoch / real_total, 1.0)
        return (1 - alpha) ** self.power