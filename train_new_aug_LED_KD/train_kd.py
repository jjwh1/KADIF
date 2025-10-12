import os
import argparse
import torch
# === DDP/분산 관련 임포트는 사용하지 않음 ===
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from tqdm import tqdm
from DDRNet import DDRNet
from DCSFKDLoss import DCSFKDLoss
from OutputKDLoss import OutputKDLoss
from DDRNet_39 import *
from functions import *
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import math
import numpy

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_parameters(model):
    print("Calculating total model parameters...")

    # 전체 모델 파라미터 수
    total_params = count_parameters(model)
    print(f"Total parameters in the model: {total_params}")

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
    TP = torch.diag(confmat)             # (K,)
    FP = confmat.sum(0) - TP             # 예측이 cls인데 정답 아님
    FN = confmat.sum(1) - TP             # 정답이 cls인데 예측 아님
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
    # ====== 단일 GPU 설정 ======
    # rank = int(os.environ["RANK"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    #
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

    val_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'val', args.scale_range, val_resize_size=(1080, 1920))
    display_dataset_info(args.dataset_dir, val_dataset)
    # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank,
    #                                  drop_last=False, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=max(1, args.batch_size//2),
                            shuffle=False, num_workers=2, pin_memory=True,
                            worker_init_fn=_seed_worker, collate_fn=collate_with_meta)

    # Model
    print(f"[Single GPU] Before model setup")
    # model = DDRNet(num_classes=args.num_classes).to(device)

    teacher_model = DualResNet(BasicBlock, [3, 4, 6, 3], num_classes=19, planes=64, spp_planes=128, head_planes=256, augment=False).to(device)
    print_model_parameters(teacher_model)
    if args.teacher_loadpath is not None:
        # map_location = {f'cuda:{0}': f'cuda:{local_rank}'}
        ckpt = torch.load(args.teacher_loadpath, map_location=device)
        teacher_model.load_state_dict(ckpt, strict=False)

    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()  # 꼭 eval 모드로 설정

    model = DDRNet(num_classes=args.num_classes).to(device)
    print_model_parameters(model)
    # model = DDP(model, device_ids=[local_rank])
    print(f"[Single GPU] Model initialized")

    # Loss, Optimizer, Scheduler
    criterion = OhemCrossEntropy(ignore_label=255)
    criterion_kd = DCSFKDLoss(loss_weight=1.0).to(device)
    criterion_output_kd = OutputKDLoss(temperature=4.0, loss_weight=0.1, ignore_index=255)

    params = list(model.parameters()) + list(criterion_kd.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs=args.epochs, warmup_epochs=10, eta_min=1e-5)
    # scheduler = WarmupPolyEpochLR(optimizer, total_epochs=args.epochs, warmup_epochs=5, warmup_ratio=5e-4)

    if args.loadpath is not None:
        # map_location = {f'cuda:{0}': f'cuda:{local_rank}'}
        state_dict = torch.load(args.loadpath, map_location=device)
        load_state_dict(model, state_dict)

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

    for epoch in range(args.epochs):
        model.train()
        criterion_kd.train()

        # train_sampler.set_epoch(epoch)
        total_loss = 0.0
        ce_loss_sum = 0.0
        kd_loss_sum = 0.0
        kd_output_loss_sum = 0.0

        num_steps = 0  # 에폭 내 배치 수

        aug_counts_local = torch.zeros(len(AUG_NAMES), device=device, dtype=torch.long)

        # if local_rank == 0:
        loop = tqdm(train_loader, desc=f"[Train] Epoch [{epoch + 1}/{args.epochs}]", ncols=110)
        # else:
        #     loop = train_loader

        for i, (imgs, labels, metas) in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs, features = model(imgs)
            with torch.no_grad():
                outputs_t, features_t = teacher_model(imgs)  # teacher forward (eval)

            # print("Student feature shape:", features.shape)
            # print("Teacher feature shape:", features_t.shape)

            loss_ce = criterion(outputs, labels)
            loss_kd_feat = criterion_kd(features, features_t)  # 기존 feature KD
            loss_kd_output = criterion_output_kd(outputs, outputs_t, labels=labels)  # 새로 추가된 output KD

            ce_loss_sum += loss_ce.item()
            kd_loss_sum += loss_kd_feat.item()
            kd_output_loss_sum += loss_kd_output.item()

            loss = loss_ce + 0.5*loss_kd_feat + 0.5*loss_kd_output
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

            total_loss += loss.item()
            num_steps += 1

            loop.set_postfix(loss=loss.item(),
                             avg_loss=total_loss / max(1, num_steps),
                             lr=scheduler.get_last_lr()[0])

        torch.cuda.empty_cache()
        # dist.barrier()
        scheduler.step()

        # ------ Train epoch 평균(단일 GPU) ------
        train_ce_loss_epoch = (ce_loss_sum / max(1, num_steps))
        train_kd_loss_epoch = (kd_loss_sum / max(1, num_steps))
        train_kd_output_loss_epoch = (kd_output_loss_sum / max(1, num_steps))
        train_loss_epoch = (total_loss / max(1, num_steps))

        # ===== Validation =====
        model.eval()
        criterion_kd.eval()

        val_ce_loss_sum = 0.0
        val_kd_loss_sum = 0.0
        val_kd_output_loss_sum = 0.0
        val_loss_sum = 0.0
        val_batches = 0.0

        # evaluation.py 의미와 맞추기 위해 int64 혼동행렬 사용
        confmat = torch.zeros((args.num_classes, args.num_classes), device=device, dtype=torch.int64)

        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"[Validate]", ncols=110)
            for imgs, labels, metas in val_iter:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                logits, features = model(imgs)
                outputs_t, features_t = teacher_model(imgs)

                vloss_ce = criterion(logits, labels)
                vloss_kd = criterion_kd(features, features_t)
                vloss_kd_output = criterion_output_kd(logits, outputs_t, labels=labels)

                val_ce_loss_sum += vloss_ce.item()
                val_kd_loss_sum += vloss_kd.item()
                val_kd_output_loss_sum += vloss_kd_output.item()

                vloss = vloss_ce + 0.5* vloss_kd + 0.5*vloss_kd_output

                val_loss_sum += vloss
                val_batches += 1.0

                preds = torch.argmax(logits, dim=1)
                confmat = _update_confmat(confmat, preds, labels, args.num_classes, ignore_index=255)

        # ---- 단일 GPU 집계 ----
        val_loss_epoch = float(val_loss_sum / max(1.0, val_batches))
        val_ce_loss_epoch = float(val_ce_loss_sum / max(1.0, val_batches))
        val_kd_loss_epoch = float(val_kd_loss_sum / max(1.0, val_batches))
        val_kd_output_loss_epoch = float(val_kd_output_loss_sum / max(1.0, val_batches))
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
            writer.add_scalar("train/loss_ce", train_ce_loss_epoch, epoch + 1)
            writer.add_scalar("train/loss_kd", train_kd_loss_epoch, epoch + 1)
            writer.add_scalar("train/loss_kd_output", train_kd_output_loss_epoch, epoch + 1)
            writer.add_scalar("train/loss_total", train_loss_epoch, epoch + 1)
            writer.add_scalar("val/loss_ce", val_ce_loss_epoch, epoch + 1)
            writer.add_scalar("val/loss_kd", val_kd_loss_epoch, epoch + 1)
            writer.add_scalar("val/loss_kd_output", val_kd_output_loss_epoch, epoch + 1)
            writer.add_scalar("val/loss_total", val_loss_epoch, epoch + 1)
            writer.add_scalar("val/mIoU", miou, epoch + 1)
            writer.add_scalar("val/Acc", acc, epoch + 1)
            writer.add_scalar("train/lr_epoch", lr, epoch + 1)

            for c, iou_c in enumerate(iou_list):
                if not math.isnan(iou_c):
                    writer.add_scalar(f"val/IoU_cls/{c}", iou_c, epoch + 1)

            for i, n in enumerate(AUG_NAMES):
                writer.add_scalar(f"aug/count/{n}", int(aug_counts[i].item()), epoch + 1)

        with open(log_path, "a") as f:
            f.write("\n%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.8f" %
                    (epoch + 1, train_loss_epoch, train_kd_output_loss_epoch,
                     val_loss_epoch, val_kd_output_loss_epoch, miou, acc, lr))

        # 베스트(Val 기준)
        if (miou > best_miou + eps) or (abs(miou - best_miou) <= eps and (epoch + 1) > 0):
            best_miou = miou
            best_epoch = epoch + 1
            # mIoU를 파일명에 포함
            ckpf = os.path.join(args.result_dir, f"model_best_e{best_epoch}_miou{best_miou:.4f}.pth")
            torch.save(_get_state_dict(model), ckpf)
            # 항상 최신 베스트를 가리키는 포인터(복사본)
            torch.save(_get_state_dict(model), os.path.join(args.result_dir, "model_best.pth"))
            torch.save(criterion_kd.state_dict(), os.path.join(args.result_dir, "kd_loss_best.pth"))

        # dist.barrier()  # 싱글 GPU에선 불필요

    if writer is not None:
        writer.close()

    # dist.destroy_process_group()  # 싱글 GPU에선 불필요


# ---------- Argparse ----------
if __name__ == "__main__":
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_P2P_DISABLE"] = "1" 
    # os.environ["NCCL_IB_DISABLE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str,  help="Path to dataset root",
                        default="/content/dataset")
    parser.add_argument("--loadpath", type=str,  help="Path to dataset root", 
                        default="/content/drive/MyDrive/KADIF/pretrained/DDRNet23s_imagenet.pth")    # "ex: ./pths/DDRNet23s_imagenet.pth"
    parser.add_argument("--teacher_loadpath", type=str, help="Path to dataset root",
                        default="/content/drive/MyDrive/KADIF/pretrained/teacher_weights_only_2_DDRNet39.pth")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--result_dir", type=str, default="/content/drive/MyDrive/KADIF/result/DDRNet_KD_1")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--crop_size", default=[1024, 1024], type=arg_as_list, help="crop size (H W)")
    parser.add_argument("--scale_range", default=[0.75, 1.25], type=arg_as_list,  help="resize Input")
    parser.add_argument("--normal_aug_prob", type=float, default=0.8, help="normal 이미지에 degradation 조합을 적용할 확률")
    parser.add_argument("--severity_min", type=int, default=1)
    parser.add_argument("--severity_max", type=int, default=5)
    
    args = parser.parse_args()
    
    print(f'Initial learning rate: {args.lr}')
    print(f'Total epochs: {args.epochs}')
    print(f'dataset path: {args.dataset_dir}')
                  
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    # torch.multiprocessing.set_start_method('spawn', force=True)  # 싱글 GPU에선 필요 없음(있어도 무방)
    train(args)
