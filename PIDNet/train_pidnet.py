import os
import argparse
import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from tqdm import tqdm
from functions_ori_edge import *
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import math
import models
from utils.utils import FullModel
from collections import OrderedDict


def _update_confmat(confmat, preds, targets, num_classes, ignore_index=255):
    valid = (targets != ignore_index)
    if not valid.any():
        return confmat
    t = targets[valid].view(-1)
    p = preds[valid].view(-1)
    k = t * num_classes + p
    hist = torch.bincount(k, minlength=num_classes * num_classes)
    hist = hist.view(num_classes, num_classes).to(confmat.device)
    confmat += hist.to(dtype=confmat.dtype)
    return confmat


def compute_miou_from_confmat(confmat):
    confmat = confmat.to(torch.float64)
    TP = torch.diag(confmat)
    FP = confmat.sum(0) - TP
    FN = confmat.sum(1) - TP
    denom = TP + FP + FN
    ious = torch.where(denom > 0, TP / denom.clamp(min=1), torch.full_like(TP, float('nan')))
    miou = torch.nanmean(ious)
    iou_list = [float(v) if not torch.isnan(v) else float('nan') for v in ious]
    return float(miou), iou_list


def compute_pixel_accuracy_from_confmat(confmat):
    total = confmat.sum().clamp(min=1)
    correct = torch.trace(confmat)
    return float((correct / total).item())


def _seed_worker(worker_id):
    import random, numpy as np, torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def train(args):
    # ----- Single GPU -----
    # rank = int(os.environ["RANK"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])

    # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0  # for logging branches

    AUG_NAMES = ["haze", "rain", "raindrop", "low_light", "overbright"]
    name_to_idx = {n: i for i, n in enumerate(AUG_NAMES)}

    # -------------------- Dataset & Dataloader --------------------
    train_dataset = SegmentationDataset(
        args.dataset_dir, args.crop_size, 'train', args.scale_range,
        val_resize_size=(1080, 1920),
        normal_aug_prob=args.normal_aug_prob,
        severity_range=(args.severity_min, args.severity_max),
    )
    display_dataset_info(args.dataset_dir, train_dataset)
    # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank,
    #                                    drop_last=True, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True,
                              worker_init_fn=_seed_worker, collate_fn=collate_with_meta)

    val_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'val', args.scale_range,
                                      val_resize_size=(1080, 1920))
    display_dataset_info(args.dataset_dir, val_dataset)
    # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank,
    #                                  drop_last=False, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=max(1, args.batch_size//2),
                            shuffle=False, num_workers=2, pin_memory=True,
                            worker_init_fn=_seed_worker, collate_fn=collate_with_meta)
    
    # loss
    criterion = CrossEntropy(ignore_label=255)
    bd_criterion = BondaryLoss()
    
    # Model
    print(f"[Single GPU] Before model setup")
    # model_base = models.pidnet.get_seg_model(num_classes=args.num_classes, load_path=args.loadpath)
    if args.loadpath is not None:
        model_base = models.pidnet.get_seg_model(num_classes=args.num_classes, load_path=args.loadpath)
    else:
        model_base = models.pidnet.get_seg_model(num_classes=args.num_classes)

    model = FullModel(model_base, criterion, bd_criterion).to(device)
    # model = DDP(model, device_ids=[local_rank])
    print(f"[Single GPU] Model initialized")

    # Optimizer, Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs=args.epochs, warmup_epochs=10, eta_min=1e-5)
    # scheduler = WarmupPolyEpochLR(optimizer, total_epochs=args.epochs, warmup_epochs=5, warmup_ratio=5e-4)

    # -------------------- Logging/TensorBoard --------------------
    writer = None
    os.makedirs(args.result_dir, exist_ok=True)
    log_path = os.path.join(args.result_dir, "log.txt")
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("Epoch\tTrain-loss\tVal-loss\tmIoU\tAcc\tlearningRate\n")
    if local_rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.result_dir, "tb"))

    def _get_state_dict(m):
        # return m.module.state_dict() if isinstance(m, DDP) else m.state_dict()
        return m.state_dict()

    # ---------- NEW: prefix 정규화 유틸 ----------
    

    def _strip_prefixes(sd, prefixes):
        new_sd = OrderedDict()
        for k, v in sd.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            new_sd[nk] = v
        return new_sd


    def _best_load_into_base(base, state):
        """
        base: 실제 모델 모듈 (ex. FullModel 내부의 base model)
        state: checkpoint state_dict (as-is).  {'state_dict': ...} 형태여도 처리함.

        - 여러 prefix 제거 전략을 시도해 가장 'missing 키 수'가 적은 로드를 선택
        - 최종 적용 후 로드 퍼센트([Info] Loaded X/Y (...%))를 함께 출력
        반환: (missing_keys, unexpected_keys)
        """
        # ckpt가 {'state_dict': ...} 형태로 저장된 경우 보정
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        strategies = [
            ("as-is",        state),
            ("strip-mm",     _strip_prefixes(state, ["module.model.", "module.", "model."])),
            ("strip-m",      _strip_prefixes(state, ["module.", "model."])),
            ("strip-model",  _strip_prefixes(state, ["model."])),
        ]

        best_name = None
        best_sd = None
        best_missing = None
        best_unexpected = None
        best_loaded_ratio = -1.0

        # 우선 평가(실제 적용은 strict=False로 '가상 로드' → 최종 한 번 더 로드)
        for name, sd_try in strategies:
            try:
                missing, unexpected = base.load_state_dict(sd_try, strict=False)

                miss_n = len(missing)
                # 로드된 키 비율(이름 일치 기준)도 보조 척도로 사용
                model_keys = set(base.state_dict().keys())
                state_keys = set(sd_try.keys())
                loaded = len(model_keys & state_keys)
                total = len(model_keys)
                ratio = (loaded / total) if total > 0 else 0.0

                if (best_missing is None) or (miss_n < len(best_missing)) or \
                (miss_n == len(best_missing) and ratio > best_loaded_ratio):
                    best_name = name
                    best_sd = sd_try
                    best_missing = missing
                    best_unexpected = unexpected
                    best_loaded_ratio = ratio
            except Exception:
                # 시도 실패 전략은 건너뜀
                continue

        # 최종 적용
        if best_sd is None:
            # 모든 전략 실패 시 as-is로 시도
            missing, unexpected = base.load_state_dict(state, strict=False)
            model_keys = set(base.state_dict().keys())
            state_keys = set(state.keys()) if isinstance(state, dict) else set()
            loaded = len(model_keys & state_keys)
            total = len(model_keys)
            print(f"[Resume] load strategy: fallback-as-is | missing={len(missing)}, unexpected={len(unexpected)}")
            print(f"[Info] Loaded {loaded}/{total} state_dict entries ({100.0 * (loaded / max(total, 1)):.2f}%) from checkpoint.")
            return missing, unexpected

        # 베스트 전략으로 실제 재적용(가중치 확정)
        missing, unexpected = base.load_state_dict(best_sd, strict=False)

        model_keys = set(base.state_dict().keys())
        state_keys = set(best_sd.keys())
        loaded = len(model_keys & state_keys)
        total = len(model_keys)

        print(f"[Resume] load strategy: {best_name} | missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"[Resume] Missing keys (first 10): {list(missing)[:10]}")
        if unexpected:
            print(f"[Resume] Unexpected keys (first 10): {list(unexpected)[:10]}")
        print(f"[Info] Loaded {loaded}/{total} state_dict entries ({100.0 * (loaded / max(total, 1)):.2f}%) from checkpoint.")

        return missing, unexpected


    def _load_model_state(m, state):
        target = m  # FullModel (no DDP)
        base = getattr(target, "model", None)
        if base is None:
            missing, unexpected = target.load_state_dict(state, strict=False)
            if local_rank == 0:
                print(f"[Resume] loaded into FullModel | missing={len(missing)}, unexpected={len(unexpected)}")
            return
        missing, unexpected = _best_load_into_base(base, state)
        if local_rank == 0:
            if missing:
                print(f"[Resume] Missing keys: {len(missing)} (showing 10) -> {missing[:10]}")
            if unexpected:
                print(f"[Resume] Unexpected keys: {len(unexpected)} (showing 10) -> {unexpected[:10]}")

    def _is_full_checkpoint(obj):
        return isinstance(obj, dict) and "model" in obj

    # -------------------- Resume / Load --------------------
    start_epoch = 0
    best_miou = float("-inf")

    def _try_read_epoch_from_last(result_dir):
        last_path = os.path.join(result_dir, "last.pth.tar")
        if os.path.isfile(last_path):
            obj = torch.load(last_path, map_location=device)
            if isinstance(obj, dict) and "epoch" in obj:
                return int(obj["epoch"])
        return None

    def _try_read_epoch_from_log(path):
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if len(lines) <= 1:
                return None
            last_line = lines[-1]
            ep_str = last_line.split()[0]
            return int(ep_str)
        except Exception:
            return None

    if args.resume is not None and os.path.isfile(args.resume):
        map_location = device
        ckpt = torch.load(args.resume, map_location=map_location)
        if _is_full_checkpoint(ckpt):
            _load_model_state(model, ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_miou  = float(ckpt.get("best_miou", float("-inf")))
            if local_rank == 0:
                print(f"[Resume: full] {args.resume} (next_epoch_idx={start_epoch}, best_mIoU={best_miou:.4f})")
        else:
            _load_model_state(model, ckpt)
            if local_rank == 0:
                print(f"[Resume: weights-only] {args.resume} → model weights loaded")

            last_path = os.path.join(args.result_dir, "last.pth.tar")
            if os.path.isfile(last_path):
                last_obj = torch.load(last_path, map_location=map_location)
                if isinstance(last_obj, dict):
                    if "optimizer" in last_obj:
                        optimizer.load_state_dict(last_obj["optimizer"])
                    if "scheduler" in last_obj:
                        scheduler.load_state_dict(last_obj["scheduler"])
                    if "epoch" in last_obj:
                        start_epoch = int(last_obj["epoch"])
                    best_miou = float(last_obj.get("best_miou", float("-inf")))
                    if local_rank == 0:
                        print(f"[Resume: pulled opt/sched] from {last_path} (next_epoch_idx={start_epoch})")

            if start_epoch == 0:
                guessed = _try_read_epoch_from_log(log_path)
                if guessed is None and args.resume_epoch is not None:
                    guessed = int(args.resume_epoch)
                if guessed is not None:
                    start_epoch = int(guessed)
                    if local_rank == 0:
                        print(f"[Resume: inferred] start_epoch set to {start_epoch} from history")

            scheduler.last_epoch = start_epoch - 1
    else:
        pass

    eps = 1e-6

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # train_sampler.set_epoch(epoch)
        total_loss = 0.0
        num_steps = 0

        aug_counts_local = torch.zeros(len(AUG_NAMES), device=device, dtype=torch.long)

        if local_rank == 0:
            loop = tqdm(train_loader, desc=f"[Train] Epoch [{epoch + 1}/{args.epochs}]", ncols=110)
        else:
            loop = train_loader

        for i, (imgs, labels, metas, edges) in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bd_gts = edges.to(device, non_blocking=True)
            
            loss, outputs = model(imgs, labels, bd_gts)
            loss.backward()
            optimizer.step()

            if isinstance(metas, (list, tuple)):
                for m in metas:
                    for (name, sev) in m.get("applied", []):
                        idx = name_to_idx.get(name)
                        if idx is not None:
                            aug_counts_local[idx] += 1
            elif isinstance(metas, dict):
                for applied in metas.get("applied", []):
                    for (name, sev) in applied:
                        idx = name_to_idx.get(name)
                        if idx is not None:
                            aug_counts_local[idx] += 1

            total_loss += loss.item()
            num_steps += 1

            if local_rank == 0:
                loop.set_postfix(loss=loss.item(),
                                 avg_loss=total_loss / max(1, num_steps),
                                 lr=scheduler.get_last_lr()[0])

        torch.cuda.empty_cache()
        # dist.barrier()
        scheduler.step()

        # ------ Train epoch 평균 (Single) ------
        train_loss_epoch = (total_loss / max(1, num_steps))

        # ===== Validation =====
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0.0
        confmat = torch.zeros((args.num_classes, args.num_classes), device=device, dtype=torch.int64)

        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"[Validate]", ncols=110) if local_rank == 0 else val_loader
            for imgs, labels, metas, edges in val_iter:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                bd_gts = edges.to(device, non_blocking=True)
                
                vloss, logits = model(imgs, labels, bd_gts)
                val_loss_sum += float(vloss.item())
                val_batches += 1.0

                preds = torch.argmax(logits[-1], dim=1)
                confmat = _update_confmat(confmat, preds, labels, args.num_classes, ignore_index=255)

        val_loss_epoch = (val_loss_sum / max(1.0, val_batches))
        miou, iou_list = compute_miou_from_confmat(confmat)
        acc = compute_pixel_accuracy_from_confmat(confmat)

        aug_counts = aug_counts_local.clone()

        # ===== Logging / Checkpoint on rank0 =====
        if local_rank == 0:
            lr_vals = scheduler.get_last_lr()
            lr = sum(lr_vals) / len(lr_vals)

            counts_str = ", ".join(f"{n}:{int(aug_counts[i].item())}" for i, n in enumerate(AUG_NAMES))
            print(f"[Epoch {epoch + 1}] Aug Applied Counts -> {counts_str}")

            if writer is not None:
                step = epoch + 1
                writer.add_scalar("train/loss", train_loss_epoch, step)
                writer.add_scalar("val/loss",   val_loss_epoch,   step)
                writer.add_scalar("val/mIoU",   miou,             step)
                writer.add_scalar("val/Acc",    acc,              step)
                writer.add_scalar("train/lr_epoch", lr,           step)
                for c, iou_c in enumerate(iou_list):
                    if not math.isnan(iou_c):
                        writer.add_scalar(f"val/IoU_cls/{c}", iou_c, step)
                for i, n in enumerate(AUG_NAMES):
                    writer.add_scalar(f"aug/count/{n}", int(aug_counts[i].item()), step)

            with open(log_path, "a") as f:
                f.write("\n%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.8f" %
                        (epoch + 1, train_loss_epoch, val_loss_epoch, miou, acc, lr))

            # ---- Save checkpoints ----
            def save_ckpt(tag_path):
                ckpt = {
                    "model": _get_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_miou": best_miou
                }
                torch.save(ckpt, tag_path)

            save_ckpt(os.path.join(args.result_dir, "last.pth.tar"))

            if (miou > best_miou + eps) or (abs(miou - best_miou) <= eps and (epoch + 1) > 0):
                best_miou  = miou
                best_epoch = epoch + 1
                torch.save(_get_state_dict(model), os.path.join(args.result_dir, "model_best.pth"))
                torch.save(_get_state_dict(model), os.path.join(args.result_dir, f"model_best_e{best_epoch}_miou{best_miou:.4f}.pth"))
                save_ckpt(os.path.join(args.result_dir, "best.pth.tar"))

        # dist.barrier()

    if local_rank == 0 and writer is not None:
        writer.close()

    # dist.destroy_process_group()


# ---------- Argparse ----------
if __name__ == "__main__":
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_P2P_DISABLE"] = "1" 
    # os.environ["NCCL_IB_DISABLE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str,  help="Path to dataset root",
                        default="/content/dataset")
    # parser.add_argument("--loadpath", type=str,  help="Path to dataset root", 
    #                     default="/content/drive/MyDrive/KADIF/pretrained/PIDNet_S_ImageNet.pth.tar")
    parser.add_argument("--loadpath", type=str,  help="Path to dataset root", 
                    default=None)
    parser.add_argument("--resume", type=str,
                        default="/content/drive/MyDrive/KADIF/result/PIDNet_s_1/last.pth.tar",
                        help="통합 ckpt(.pth/.tar) 또는 가중치(.pth) 경로")
    parser.add_argument("--resume_epoch", type=int, default=None,
                        help="weights-only 재개 시 마지막 완료 epoch(1-index)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--result_dir", type=str, default="/content/drive/MyDrive/KADIF/result/PIDNet_s_1_2")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--crop_size", default=[1024, 1920], type=arg_as_list, help="crop size (H W)")
    parser.add_argument("--scale_range", default=[0.75, 1.25], type=arg_as_list,  help="resize Input")
    parser.add_argument("--normal_aug_prob", type=float, default=0.5, help="normal 이미지에 degradation 조합을 적용할 확률")
    parser.add_argument("--severity_min", type=int, default=1)
    parser.add_argument("--severity_max", type=int, default=5)
    
    args = parser.parse_args()
    print(f'Initial learning rate: {args.lr}')
    print(f'Total epochs: {args.epochs}')
    print(f'dataset path: {args.dataset_dir}')
                  
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.set_start_method('spawn', force=True)
    train(args)
