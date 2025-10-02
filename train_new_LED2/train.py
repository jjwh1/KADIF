import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from tqdm import tqdm
from DDRNet import DDRNet
from functions import *
import math
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


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


def train(args):
    # 단일 GPU 모드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- Dataset & Dataloader --------------------
    train_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'train', args.scale_range)
    display_dataset_info(args.dataset_dir, train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    # (검증용) val dataset_dir 업데이트 해야함!!!
    val_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'val', args.scale_range)
    display_dataset_info(args.dataset_dir, val_dataset)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model
    print(f"[Single GPU] Before model setup")
    model = DDRNet(num_classes=args.num_classes).to(device)
    print(f"[Single GPU] Model initialized")

    # Loss, Optimizer, Scheduler
        # focal loss 사용 :클래스 가중치 계산
    # 클래스 가중치 불러오기 (없으면 계산해서 저장)

    # class_weights = compute_or_load_class_weights(
    #     train_loader, args.num_classes,
    #     cache_path=args.class_weights_dir,
    #     method="inverse"   # inverse, effective_num
    # ).to(device)
    # criterion = FocalLoss(gamma=2.0, alpha=class_weights, ignore_index=255)


    weight = torch.ones(19)              # 기본은 전부 1.0
    weight[1] = 7.0
    weight[2] = 7.0
    weight[3] = 7.0
    weight[4] = 7.0
    weight[5] = 6.7
    weight[7] = 6.5
    weight[8] = 6.5     
    weight[9] = 9
    weight[10] = 10
    weight[11] = 6.5   
    weight[12] = 9
    weight[13] = 6.5
    weight[16] = 6.5
    weight[17] = 10           

    criterion = CrossEntropy(ignore_label=255, weight=weight.to(device))
    # criterion = CrossEntropy(ignore_label=255)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)
#     scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs=args.epochs, warmup_epochs=10, eta_min=1e-5)
    scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs=args.epochs, warmup_epochs=10, eta_min=1e-5)

    ## pretrained가져오거나 none 일때
    # if args.loadpath is not None:
    #     state_dict = torch.load(args.loadpath, map_location=device)
    #     load_state_dict(model, state_dict)
    # start_epoch=0

    ## 학습 끊겨 checkpoint 불러올 때
    if args.loadpath is not None:
        ckpt = torch.load(args.loadpath, map_location=device, weights_only=False)
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
    os.makedirs(args.result_dir, exist_ok=True)
    log_path = os.path.join(args.result_dir, "log.txt")
    with open(log_path, 'w') as f:
        f.write("Epoch\t\tTrain-loss\t\tVal-loss\t\tmIoU\t\tAcc\t\tlearningRate\n")
    writer = SummaryWriter(log_dir=os.path.join(args.result_dir, "board"))

    def _get_state_dict(m):
        return m.state_dict()

    best_miou = float("-inf")
    eps = 1e-6

    for epoch in range(start_epoch,args.epochs):
        model.train()
        total_loss = 0.0
        num_steps = 0  # 에폭 내 배치 수

        loop = tqdm(train_loader, desc=f"[Train] Epoch [{epoch + 1}/{args.epochs}]", ncols=110)

        for i, (imgs, labels) in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)

            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            total_loss += loss.item()
            num_steps += 1

            loop.set_postfix(loss=loss.item(),
                             avg_loss=total_loss / max(1, num_steps),
                             lr=scheduler.get_last_lr()[0])

        torch.cuda.empty_cache()
        scheduler.step()

        # ------ Train epoch 평균 ------
        train_loss_epoch = (total_loss / max(1, num_steps))

        # ===== Validation =====
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0.0

        confmat = torch.zeros((args.num_classes, args.num_classes), device=device, dtype=torch.int64)

        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"[Validate]", ncols=110)
            for imgs, labels in val_iter:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                logits = model(imgs)
                vloss = criterion(logits, labels)
                val_loss_sum += vloss.item()
                val_batches += 1.0

                preds = torch.argmax(logits, dim=1)
                confmat = _update_confmat(confmat, preds, labels, args.num_classes, ignore_index=255)

        val_loss_epoch = (val_loss_sum / max(1, val_batches))
        miou, iou_list = compute_miou_from_confmat(confmat)
        acc = compute_pixel_accuracy_from_confmat(confmat)

        # ===== Logging / Checkpoint =====
        lr = scheduler.get_last_lr()
        lr = sum(lr) / len(lr)

        if writer is not None:
            writer.add_scalar("train/loss", train_loss_epoch, epoch + 1)
            writer.add_scalar("val/loss", val_loss_epoch, epoch + 1)
            writer.add_scalar("val/mIoU", miou, epoch + 1)
            writer.add_scalar("val/Acc", acc, epoch + 1)
            writer.add_scalar("train/lr_epoch", lr, epoch + 1)

            for c, iou_c in enumerate(iou_list):
                if not math.isnan(iou_c):
                    writer.add_scalar(f"val/IoU_cls/{c}", iou_c, epoch + 1)

            # (선택) 히스토그램으로 한 번에 보기
            import torch as _torch
            writer.add_histogram(
                "val/per_class_IoU_hist",
                _torch.tensor([0.0 if math.isnan(x) else x for x in iou_list]),
                epoch + 1
            )

        with open(log_path, "a") as f:
            f.write("\n%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.8f" %
                    (epoch + 1, train_loss_epoch, val_loss_epoch, miou, acc, lr))

        # 베스트(Val Loss 기준)
        # if (miou > best_miou + eps) or (abs(miou - best_miou) <= eps and (epoch + 1) > 0):
        #     best_miou = miou
        #     best_epoch = epoch + 1
        #     ckpf = os.path.join(args.result_dir, f"model_best_e{best_epoch}_miou{best_miou:.4f}.pth")
        #     torch.save(_get_state_dict(model), ckpf)
        #     torch.save(_get_state_dict(model), os.path.join(args.result_dir, "model_best.pth"))
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


# ---------- Argparse ----------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_dir", type=str, help="Path to dataset root",
#                         default=r"C:\Users\8138\Desktop\KADIF\seg\SemanticDataset_trainvalid")
#     parser.add_argument("--loadpath", type=str, help="Path to pretrained model",
#                         default=None)
#     parser.add_argument("--epochs", type=int, default=500)
#     parser.add_argument("--result_dir", type=str, default=r"D:\KADIF")
#     parser.add_argument("--lr", type=float, default=1e-2)
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--num_classes", type=int, default=19)
#     parser.add_argument("--crop_size", default=[1024, 1024], type=arg_as_list, help="crop size (H W)")
#     parser.add_argument("--scale_range", default=[0.75, 1.25], type=arg_as_list, help="resize Input")
#
#     args = parser.parse_args()
#
#     print(f'Initial learning rate: {args.lr}')
#     print(f'Total epochs: {args.epochs}')
#     print(f'dataset path: {args.dataset_dir}')
#
#     result_dir = Path(args.result_dir)
#     result_dir.mkdir(parents=True, exist_ok=True)
#     train(args)


# ---------- Argparse ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset root",
                        default="/content/dataset")    # -v /mnt/c/Users/8138/Desktop/KADIF/seg/SemanticDataset_trainvalid:/workspace/dataset \
    parser.add_argument("--loadpath", type=str, help="Path to pretrained model",
                        default="/content/drive/MyDrive/KADIF/result/DDRNet_8/checkpoint_latest.pth")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--result_dir", type=str, default="/content/drive/MyDrive/KADIF/result/DDRNet_8_2")   # -v /mnt/d/KADIF:/workspace/result \
    parser.add_argument("--class_weights_dir", type=str, default="/content/drive/MyDrive/KADIF/class_weights.pt",
                help="focal loss 사용시알파 계산을 위한 trainset의 class weights")  # -v /mnt/c/Users/8138/Desktop/KADIF/seg/SemanticDataset_trainvalid:/workspace/dataset \
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--crop_size", default=[1024, 1024], type=arg_as_list, help="crop size (H W)")
    parser.add_argument("--scale_range", default=[0.75, 1.25], type=arg_as_list, help="resize Input")

    args = parser.parse_args()

    print(f'Initial learning rate: {args.lr}')
    print(f'Total epochs: {args.epochs}')
    print(f'dataset path: {args.dataset_dir}')

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    train(args)
