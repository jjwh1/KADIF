import torch
import torch.nn as nn
import torch.nn.functional as F

class OutputKDLoss(nn.Module):
    def __init__(self, temperature=4.0, loss_weight=1.0, ignore_index=None):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, student_logits, teacher_logits, labels=None):
        T = self.temperature

        if isinstance(student_logits, (tuple, list)):
            student_logits = student_logits[0] + 0.4 * student_logits[1]
        if isinstance(teacher_logits, (tuple, list)):
            teacher_logits = teacher_logits[0] + 0.4 * teacher_logits[1]

        B, C, H, W = student_logits.shape

        s_logp = F.log_softmax(student_logits / T, dim=1)      # [B,C,H,W]
        t_prob = F.softmax(teacher_logits / T, dim=1).detach() # [B,C,H,W]

        # ---- valid mask 만들기 (labels가 Tensor일 때만 ignore_index 적용) ----
        if isinstance(labels, torch.Tensor) and self.ignore_index is not None:
            # [B,H,W] -> [B,1,H,W]
            valid = labels.ne(self.ignore_index).unsqueeze(1)
        else:
            # 전 픽셀 사용
            valid = torch.ones((B, 1, H, W), dtype=torch.bool, device=student_logits.device)

        # 마스크 적용
        s_logp = s_logp.masked_fill(~valid, 0.0)
        t_prob = t_prob.masked_fill(~valid, 0.0)

        # KL per-pixel: 채널 합, 나머지는 평균
        kl = F.kl_div(s_logp, t_prob, reduction='none')  # [B,C,H,W]
        kl = kl.sum(dim=1)                               # [B,H,W]

        num_valid = valid.sum().clamp_min(1)
        loss = kl.sum() / num_valid                      # 유효 픽셀 평균
        loss = loss * (T ** 2)

        return self.loss_weight * loss
