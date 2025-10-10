import os
import argparse
import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from DDRNet_2 import DDRNet
from functions import *
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import math


def _update_confmat(confmat, preds, targets, num_classes, ignore_index=255):
    # preds: (B,H,W) argmax 결과, targets: (B,H,W)
    valid = (targets != ignore_index)
    if not valid.any():
        return confmat

    t = targets[valid].view(-1)
    p = preds[valid].view(-1)
    k = t * num_classes + p  # (t,p) 쌍을 1D 인덱스로
    hist = torch.bincount(k, minlength=num_classes * num_classes)
    hist = hist.view(num_classes, num_classes).to(confmat.device)
    confmat += hist.to(dtype=confmat.dtype)
    return confmat


def compute_miou_from_confmat(confmat):
    """
    evaluation.py의 compute_miou와 동일한 정의:
    IoU(cls) = TP / (TP + FP + FN), 분모=0이면 NaN, mIoU는 NaN 무시 평균
    """
    confmat = confmat.to(torch.float64)  # 안전한 정밀도
    TP = torch.diag(confmat)  # (K,)
    FP = confmat.sum(0) - TP  # 예측이 cls인데 정답 아님
    FN = confmat.sum(1) - TP  # 정답이 cls인데 예측 아님
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
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def train(args):
    # rank = int(os.environ["RANK"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])

    # dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    AUG_NAMES = ["haze", "rain", "raindrop", "low_light", "overbright"]
    name_to_idx = {n: i for i, n in enumerate(AUG_NAMES)}

    # -------------------- Dataset & Dataloader --------------------
    train_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'train', args.scale_range,
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

    # (검증용) val dataset_dir 업데이트 해야함!!!
    val_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'val', args.scale_range,
                                      val_resize_size=(1080, 1920))
    display_dataset_info(args.dataset_dir, val_dataset)
    # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank,
    #                                  drop_last=False, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=max(1, args.batch_size // 2),
                            shuffle=False, num_workers=2, pin_memory=True,
                            worker_init_fn=_seed_worker, collate_fn=collate_with_meta)

    # Model
    print(f"[Single GPU] Before model setup")
    model = DDRNet(num_classes=args.num_classes).to(device)
    # model = DDP(model, device_ids=[local_rank])
    print(f"[Single GPU] Model initialized")


    # Loss, Optimizer, Scheduler

    # focal loss 사용 :클래스 가중치 계산
    # 클래스 가중치 불러오기 (없으면 계산해서 저장)
    # class_weights = compute_or_load_class_weights(
    #     train_loader, args.num_classes,
    #     cache_path=args.class_weights_dir,
    #     method="effective_num"
    # ).to(device)
    # criterion = FocalLoss(gamma=2.0, alpha=class_weights, ignore_index=255)

    # criterion = CrossEntropy(ignore_label=255)
    criterion = OhemCrossEntropy(ignore_label=255)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs=args.epochs, warmup_epochs=10, eta_min=1e-5)
    # scheduler = WarmupPolyEpochLR(optimizer, total_epochs=args.epochs, warmup_epochs=5, warmup_ratio=5e-4)

    ## pretrained가져오거나 none 일때
    # if args.loadpath is not None:
    #     # map_location = {f'cuda:{0}': f'cuda:{local_rank }'}
    #     state_dict = torch.load(args.loadpath, map_location=device)
    #     load_state_dict(model, state_dict)
    # start_epoch=0

    ## 학습 끊겨 checkpoint 불러올 때
    if args.loadpath is not None:
        ckpt = torch.load(args.loadpath, map_location=device,weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_miou = ckpt.get("best_miou", float("-inf"))  # 혹시 저장된 값 있으면 복원
        print(f"✅ Resumed training from epoch {start_epoch}, best_miou={best_miou:.4f}")
    else:
        start_epoch = 0
        best_miou = float("-inf")


    # -------------------- Logging/TensorBoard --------------------
    writer = None
    # if local_rank == 0:
    os.makedirs(args.result_dir, exist_ok=True)
    log_path = os.path.join(args.result_dir, "log.txt")
    with open(log_path, 'w') as f:
        f.write("Epoch\t\tTrain-loss\t\tlearningRate\n")
    writer = SummaryWriter(log_dir=os.path.join(args.result_dir, "tb"))

    def _get_state_dict(m):
        # return m.module.state_dict() if isinstance(m, DDP) else m.state_dict()
        return m.state_dict()

    best_miou = float("-inf")
    eps = 1e-6

    for epoch in range(start_epoch,args.epochs):
        model.train()
        # train_sampler.set_epoch(epoch)
        total_loss = 0.0
        num_steps = 0  # 에폭 내 배치 수

        aug_counts_local = torch.zeros(len(AUG_NAMES), device=device, dtype=torch.long)

        # if local_rank == 0:
        loop = tqdm(train_loader, desc=f"[Train] Epoch [{epoch + 1}/{args.epochs}]", ncols=110)
        # else:
        #     loop = train_loader

        for i, (imgs, labels, metas) in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # === 증강 카운트 집계 ===
            if isinstance(metas, (list, tuple)):  # ✅ collate_with_meta 경로
                for m in metas:
                    for (name, sev) in m.get("applied", []):
                        idx = name_to_idx.get(name)
                        if idx is not None:
                            aug_counts_local[idx] += 1
            elif isinstance(metas, dict):  # (혹시 다른 collate를 쓰는 경우 호환)
                for applied in metas.get("applied", []):
                    for (name, sev) in applied:
                        idx = name_to_idx.get(name)
                        if idx is not None:
                            aug_counts_local[idx] += 1

            # torch.cuda.synchronize()

            total_loss += loss.item()
            num_steps += 1

            loop.set_postfix(loss=loss.item(),
                             avg_loss=total_loss / max(1, num_steps),
                             lr=scheduler.get_last_lr()[0])

        torch.cuda.empty_cache()
        # dist.barrier()
        scheduler.step()

        # ------ Train epoch 평균 ------
        train_loss_epoch = (total_loss / max(1, num_steps))

        # ===== Validation =====
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0.0

        # evaluation.py 의미와 맞추기 위해 int64 혼동행렬 사용
        confmat = torch.zeros((args.num_classes, args.num_classes), device=device, dtype=torch.int64)

        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"[Validate]", ncols=110)
            for imgs, labels, metas in val_iter:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                logits = model(imgs)
                vloss = criterion(logits, labels)
                val_loss_sum += vloss.item()
                val_batches += 1.0

                preds = torch.argmax(logits, dim=1)
                confmat = _update_confmat(confmat, preds, labels, args.num_classes, ignore_index=255)

        # ---- 집계 ----
        val_loss_epoch = (val_loss_sum / max(1, val_batches))
        miou, iou_list = compute_miou_from_confmat(confmat)
        acc = compute_pixel_accuracy_from_confmat(confmat)

        aug_counts = aug_counts_local.clone()

        # ===== Logging / Checkpoint =====
        lr = scheduler.get_last_lr()
        lr = sum(lr) / len(lr)

        counts_str = ", ".join(f"{n}:{int(aug_counts[i].item())}" for i, n in enumerate(AUG_NAMES))
        print(f"[Epoch {epoch + 1}] Aug Applied Counts -> {counts_str}")

        # ⬇️ 에폭당 한 번만 기록
        if writer is not None:
            writer.add_scalar("train/loss", train_loss_epoch, epoch + 1)
            writer.add_scalar("val/loss", val_loss_epoch, epoch + 1)
            writer.add_scalar("val/mIoU", miou, epoch + 1)
            writer.add_scalar("val/Acc", acc, epoch + 1)
            writer.add_scalar("train/lr_epoch", lr, epoch + 1)

            for c, iou_c in enumerate(iou_list):
                if not math.isnan(iou_c):
                    writer.add_scalar(f"val/IoU_cls/{c}", iou_c, epoch + 1)

            for i, n in enumerate(AUG_NAMES):
                writer.add_scalar(f"aug/count/{n}", int(aug_counts[i].item()), epoch + 1)

        with open(log_path, "a") as f:
            f.write("\n%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.8f" %
                    (epoch + 1, train_loss_epoch, val_loss_epoch, miou, acc, lr))

        # 베스트(Val Loss 기준)
        # if (miou > best_miou + eps) or (abs(miou - best_miou) <= eps and (epoch + 1) > 0):
        #     best_miou = miou
        #     best_epoch = epoch + 1
        #     # mIoU를 파일명에 포함
        #     ckpf = os.path.join(args.result_dir, f"model_best_e{best_epoch}_miou{best_miou:.4f}.pth")
        #     torch.save(_get_state_dict(model), ckpf)
        #     # 항상 최신 베스트를 가리키는 포인터(복사본)
        #     torch.save(_get_state_dict(model), os.path.join(args.result_dir, "model_best.pth"))

        # dist.barrier()  # 다음 epoch 동기화
        # ===== Save checkpoint (best or latest) =====
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": best_miou
        }

        # always save latest
        torch.save(ckpt, os.path.join(args.result_dir, "checkpoint_latest.pth"))

        # save best separately
        if (miou > best_miou + eps) or (abs(miou - best_miou) <= eps and (epoch + 1) > 0):
            best_miou = miou
            best_epoch = epoch + 1

            # best_miou 갱신 후 ckpt 다시 정의해야 함
            ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_miou": best_miou
            }

            ckpf = os.path.join(args.result_dir, f"model_best_e{best_epoch}_miou{best_miou:.4f}.pth")
            torch.save(ckpt, ckpf)
            torch.save(ckpt, os.path.join(args.result_dir, "checkpoint_best.pth"))

    if writer is not None:
        writer.close()

    # dist.destroy_process_group()


# ---------- Argparse ----------
if __name__ == "__main__":
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    # os.environ["NCCL_IB_DISABLE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset root",
                        default="/content/dataset") 
    parser.add_argument("--loadpath", type=str, help="Path to dataset root",
                        default="/content/drive/MyDrive/KADIF/result/DDRNet_12_3/checkpoint_latest.pth")  
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--result_dir", type=str, default="/content/drive/MyDrive/KADIF/result/DDRNet_12_4")  
    parser.add_argument("--class_weights_dir", type=str, default="/content/drive/MyDrive/KADIF/class_weights.pt",
                help="focal loss 사용시알파 계산을 위한 trainset의 class weights")  
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--crop_size", default=[1024, 1024], type=arg_as_list, help="crop size (H W)")
    parser.add_argument("--scale_range", default=[0.75, 1.25], type=arg_as_list, help="resize Input")
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
