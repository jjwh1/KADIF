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

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import cv2  # [NEW]
import torch.nn.functional as F 

from torch.utils.data._utils.collate import default_collate
from degra_for_aug import build_transform


# ====== Edge kernel/Pad 설정 (기억해둔 코드와 동일하게) ======
y_k_size = 6  # [NEW]
x_k_size = 6  # [NEW]


def collate_with_meta(batch):
    """
    batch: list of samples; each sample is
      (img, label, meta, edge)
    반환:
      imgs:   Tensor [B, C, H, W]
      labels: Tensor [B, H, W]
      metas:  list[dict] 길이 B
      edges:  Tensor [B, H, W]  (0/1)
    """
    imgs, labels, metas, edges = [], [], [], []  # [UPDATED]
    for sample in batch:
        # 항상 4개를 기대
        if len(sample) == 4:
            img, lab, meta, edge = sample
        elif len(sample) == 3:
            # 구버전 호환: edge가 없으면 0 텐서 채우기
            img, lab, meta = sample
            edge = torch.zeros_like(lab, dtype=torch.float32)
        else:
            img, lab = sample
            meta = {}
            edge = torch.zeros_like(lab, dtype=torch.float32)
        imgs.append(img)
        labels.append(lab)
        metas.append(meta)
        edges.append(edge)
    return torch.stack(imgs, dim=0), torch.stack(labels, dim=0), metas, torch.stack(edges, dim=0)  # [UPDATED]


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
        val_resize_size=(1080, 1920),
        normal_aug_prob=0.5,
        normal_aug_chains=None,
        severity_range=(1, 5),
        edge_size=4,            # [NEW] dilation kernel size
        edge_pad=True           # [NEW] padding 보정 on/off
    ):
        self.crop_size = crop_size  # (H, W)
        self.scale_range = scale_range
        self.is_train = is_train
        self.val_resize_size = val_resize_size

        self.mean = mean
        self.std = std

        self.bilinear = transforms.InterpolationMode.BILINEAR
        self.nearest = transforms.InterpolationMode.NEAREST

        self.normal_aug_prob = float(normal_aug_prob)
        self.severity_range = (int(severity_range[0]), int(severity_range[1]))

        default_chains = [
            ("rain", "raindrop", "low_light"),
            ("rain", "raindrop"),
            ("rain", "raindrop", "haze"),
            ("low_light",),
            ("haze",),
        ]
        self.normal_aug_chains = list(normal_aug_chains) if normal_aug_chains else default_chains

        # --- edge 관련 옵션 ---
        self.edge_size = int(edge_size)     # [NEW]
        self.edge_pad = bool(edge_pad)      # [NEW]

    # ---- 내부 유틸: 기하 증강 ----
    def _random_scale(self, image, label, edge=None):  # [UPDATED]
        s = random.uniform(self.scale_range[0], self.scale_range[1])
        w, h = image.size
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        image = TF.resize(image, (nh, nw), interpolation=self.bilinear)
        label = TF.resize(label, (nh, nw), interpolation=self.nearest)
        if edge is not None:
            edge = TF.resize(edge, (nh, nw), interpolation=self.nearest)
        return image, label, edge  # [UPDATED]

    def _pad_and_random_crop(self, image, label, edge=None):  # [UPDATED]
        H, W = self.crop_size
        _, h, w = 0, image.size[1], image.size[0]
        pad_h, pad_w = max(H - h, 0), max(W - w, 0)
        if pad_h > 0 or pad_w > 0:
            # left, top, right, bottom
            image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
            label = TF.pad(label, (0, 0, pad_w, pad_h), fill=255)
            if edge is not None:
                edge = TF.pad(edge, (0, 0, pad_w, pad_h), fill=0)
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)
        if edge is not None:
            edge = TF.crop(edge, i, j, h, w)
        return image, label, edge  # [UPDATED]

    def _random_hflip(self, image, label, edge=None, p=0.5):  # [UPDATED]
        if random.random() < p:
            image = TF.hflip(image)
            label = TF.hflip(label)
            if edge is not None:
                edge = TF.hflip(edge)
        return image, label, edge  # [UPDATED]

    def _resize_to(self, image, label, size_hw, edge=None):  # [UPDATED]
        H, W = int(size_hw[0]), int(size_hw[1])
        image = TF.resize(image, (H, W), interpolation=self.bilinear)
        label = TF.resize(label, (H, W), interpolation=self.nearest)
        if edge is not None:
            edge = TF.resize(edge, (H, W), interpolation=self.nearest)
        return image, label, edge  # [UPDATED]

    # --- 단일 증강 적용(강도 랜덤) ---
    def _apply_single_aug(self, image, aug_name):
        sev = random.randint(self.severity_range[0], self.severity_range[1])
        t = build_transform(aug_name, sev)
        return t(image), (aug_name, sev)

    # --- 체인 증강 적용 (여러 개 순차 적용) ---
    def _apply_aug_chain(self, image, chain_names):
        out = image
        applied = []
        for name in chain_names:
            out, item = self._apply_single_aug(out, name)
            applied.append(item)  # (aug_name, sev)
        return out, applied

    # --- [NEW] 라벨로부터 edge 생성 (Canny + dilation + optional pad 보정) ---
    def _make_edge_from_label(self, label_img_pil):
        """
        label_img_pil: PIL Image 'L' (하나의 채널, 클래스 인덱스 맵)
        반환: PIL Image 'L' (0/255)
        """
        lab_np = np.array(label_img_pil, dtype=np.uint8)

        # Canny (라벨 경계 검출) - 기억한 코드와 동일한 구조
        edge = cv2.Canny(lab_np, 0.1, 0.2)  # 매우 낮은 threshold (경계만)
        kernel = np.ones((self.edge_size, self.edge_size), np.uint8)

        if self.edge_pad:
            # 경계 손실 방지를 위한 크롭/패드 보정
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size, y_k_size), (x_k_size, x_k_size)), mode='constant')

        edge = (cv2.dilate(edge, kernel, iterations=1) > 50).astype(np.uint8) * 255  # 0/255
        return Image.fromarray(edge, mode='L')

    def __call__(self, image, label, tag="normal"):
        # === 엣지 생성은 "원본 라벨" 기반으로 먼저 만든 뒤 동일 기하변환을 공유 ===
        edge_img = self._make_edge_from_label(label)  # [NEW]

        # === 태그 기반 degradation 증강 ===
        tag = (tag or "normal").lower()

        if not self.is_train:
            # VALID: ToTensor + Normalize 만 (edge도 같이 리사이즈)
            if self.val_resize_size is not None:
                image, label, edge_img = self._resize_to(image, label, self.val_resize_size, edge=edge_img)  # [UPDATED]

            # ToTensor + Normalize
            img_t = TF.to_tensor(image)
            img_t = TF.normalize(img_t, self.mean, self.std)
            lab_t = torch.from_numpy(np.array(label, dtype=np.uint8)).long()

            # 엣지는 0/1 float 텐서로 반환 (BCE에 안전)
            edge_t = torch.from_numpy((np.array(edge_img, dtype=np.uint8) > 127).astype(np.float32))  # [H, W] 0/1  # [NEW]

            meta = {"tag": tag, "applied": []}
            return img_t, lab_t, meta, edge_t  # [UPDATED] 항상 4개

        # TRAIN: 기하 증강(엣지 포함)
        image, label, edge_img = self._random_scale(image, label, edge=edge_img)
        image, label, edge_img = self._pad_and_random_crop(image, label, edge=edge_img)
        image, label, edge_img = self._random_hflip(image, label, edge=edge_img, p=0.5)

        applied = []
        if tag == "normal":
            if random.random() < self.normal_aug_prob:
                chain = random.choice(self.normal_aug_chains)
                image, applied = self._apply_aug_chain(image, chain)

        # ToTensor + Normalize
        img_t = TF.to_tensor(image)
        img_t = TF.normalize(img_t, self.mean, self.std)
        lab_t = torch.from_numpy(np.array(label, dtype=np.uint8)).long()

        # 엣지 텐서 (0/1 float)
        edge_t = torch.from_numpy((np.array(edge_img, dtype=np.uint8) > 127).astype(np.float32))  # [NEW]

        meta = {"tag": tag, "applied": applied}
        return img_t, lab_t, meta, edge_t  # [UPDATED]


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
        "low_light_byLED": "low_light",
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
        val_resize_size=(1080, 1920),
        normal_aug_prob=0.5,
        severity_range=(1, 5),
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

            # --- train 세트일 때만 low_light 폴더 제외 ---
            if self.subset == "train" and "/low_light/" in p.replace("\\", "/"):
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
            val_resize_size=val_resize_size,
            normal_aug_prob=normal_aug_prob,  # ← 필요 시 조정
            severity_range=severity_range,  # ← 필요 시 조정
            normal_aug_chains=[
                ("rain", "raindrop", "low_light"),
                ("rain", "raindrop"),
                ("rain", "raindrop", "haze"),
                ("low_light",),
                ("haze",),
            ]
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
        img_t, lab_t, meta, edge_t = self.transform(img, lab, tag=tag)  # [UPDATED]
        return img_t, lab_t, meta, edge_t  # [UPDATED]


# class CrossEntropy(nn.Module):
#     def __init__(self, ignore_label= 255, weight= None, aux_weights = [1, 0.4]):
#         super().__init__()
#         self.aux_weights = aux_weights
#         self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

#     def _forward(self, preds, labels):
#         # preds in shape [B, C, H, W] and labels in shape [B, H, W]
#         return self.criterion(preds, labels)

#     def forward(self, preds, labels):
#         if isinstance(preds, tuple):
#             return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
#         return self._forward(preds, labels)    



# class CrossEntropy(nn.Module):
#     def __init__(self, ignore_label=255, weight=None):
#         super(CrossEntropy, self).__init__()
#         self.ignore_label = ignore_label
#         self.criterion = nn.CrossEntropyLoss(
#             weight=weight,
#             ignore_index=ignore_label
#         )

#     def _forward(self, score, target):

#         loss = self.criterion(score, target)

#         return loss

#     def forward(self, score, target):

#         if config.MODEL.NUM_OUTPUTS == 1:
#             score = [score]

#         balance_weights = config.LOSS.BALANCE_WEIGHTS
#         sb_weights = config.LOSS.SB_WEIGHTS
#         if len(balance_weights) == len(score):
#             return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
#         elif len(score) == 1:
#             return sb_weights * self._forward(score[0], target)
        
#         else:
#             raise ValueError("lengths of prediction and target are not identical!")


import torch
import torch.nn as nn

class CrossEntropy(nn.Module):
    """
    cfg 없이 직접 하이퍼파라미터를 주입해 쓰는 버전.
    - num_outputs: 모델 주 출력(branch) 개수. (예: 1 또는 2)
    - balance_weights: 다중 출력일 때 각 분기에 곱할 가중치 리스트/튜플.
    - sb_weights: 단일 출력일 때 곱할 스칼라 가중치.
    - ignore_label: 무시할 라벨(시티스케이프는 보통 255)
    - weight: 클래스별 가중치 텐서(optional)
    """
    def __init__(
        self,
        ignore_label: int = 255,
        weight=None,
        num_outputs: int = 2,
        balance_weights=(0.5, 0.5),
        sb_weights: float = 0.5,
    ):
        super().__init__()
        self.ignore_label = int(ignore_label)
        self.num_outputs = int(num_outputs)
        self.balance_weights = list(balance_weights) if balance_weights is not None else None
        self.sb_weights = float(sb_weights)
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_label)

    def _forward(self, score, target):
        # score: (B,C,H,W), target: (B,H,W)
        return self.criterion(score, target)

    def forward(self, score, target):
        # 점검: score가 텐서 하나인지, 리스트/튜플(보조 분기 포함)인지
        is_seq = isinstance(score, (list, tuple))

        # 모델이 “단일 출력”으로 설정된 경우
        if self.num_outputs == 1:
            # 리스트/튜플이 들어와도 첫 번째만 사용
            if is_seq:
                score = score[0]
            return self.sb_weights * self._forward(score, target)

        # 모델이 “다중 출력”으로 설정된 경우
        if not is_seq:
            # 실사용 중 모델이 학습 모드가 아닐 때 주 출력만 나오는 경우를 방어적으로 처리
            # (원하면 여기서 예외를 던지도록 바꿔도 됩니다)
            return self.sb_weights * self._forward(score, target)

        # 분기 수와 가중치 수 일치 검사
        if self.balance_weights is None or len(self.balance_weights) != len(score):
            raise ValueError(
                f"Mismatch between outputs ({len(score)}) and balance_weights "
                f"({0 if self.balance_weights is None else len(self.balance_weights)})."
            )

        # 가중합
        losses = [w * self._forward(s, target) for w, s in zip(self.balance_weights, score)]
        return sum(losses)
    
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
        


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce
        
    def forward(self, bd_pre, bd_gt):

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        
        return loss
    
if __name__ == '__main__':
    a = torch.zeros(2,64,64)
    a[:,5,:] = 1
    pre = torch.randn(2,1,16,16)
    
    Loss_fc = BondaryLoss()
    loss = Loss_fc(pre, a.to(torch.uint8))
