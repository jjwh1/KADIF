from typing import Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Assuming mmrazor.registry and MODELS are part of the framework you're using

global_image_counter = 0

class DCSFKDLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(DCSFKDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu
        self.channel_align = nn.Conv2d(256, 128, kernel_size=1)  # teacher:256 → student:128

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances."""
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def forward(self, preds_S: Union[torch.Tensor, Tuple], preds_T: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        global global_image_counter
        """Forward computation."""
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S, ), (preds_T, )

        loss = 0.0
        index = 0

        for pred_S, pred_T in zip(preds_S, preds_T):
            index += 1
            if pred_T.shape != pred_S.shape:
                pred_T = self.channel_align(pred_T)

            size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]

            if size_S != size_T:
                if self.resize_stu:
                    pred_S = F.interpolate(pred_S, size_T, mode='bilinear', align_corners=False)
                else:
                    pred_T = F.interpolate(pred_T, size_S, mode='bilinear', align_corners=False)

            assert pred_S.shape == pred_T.shape

            # Apply normalization
            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)

            # Compute global average pooling to get channel-wise attentions
            gap_T = F.adaptive_avg_pool2d(norm_T, (1, 1)).view(norm_T.size(0), -1)

            # Apply softmax to the teacher attentions and scale to the range 1-10
            softmax_att_T = F.softmax(gap_T, dim=-1) * 9 + 1  # Scale and shift

            # filename3 = 'weight_' + str(index) + str(norm_T.shape) + str(global_image_counter)
            # lowchannels, highchannels = self.visualize_weight_matrix(softmax_att_T, 'weight matrix', filename3)

            # filename1 = 'norm_T' + str(index) + str(norm_T.shape) + str(global_image_counter)
            # self.visualize_and_save(norm_T, 'norm_T', filename1, lowchannels, highchannels)

            # Apply teacher attentions to student feature maps
            att_T = softmax_att_T.view(norm_S.size(0), norm_S.size(1), 1, 1).expand_as(norm_S)

            weighted_S = norm_S * att_T

            # Flatten the feature maps for MSE computation
            weighted_S = weighted_S.view(weighted_S.size(0), weighted_S.size(1), -1)
            norm_T = norm_T.view(norm_T.size(0), norm_T.size(1), -1)
            global_image_counter += 1

            # Compute MSE loss with teacher attentions applied to student feature maps
            loss += F.mse_loss(weighted_S, norm_T)

        return loss * self.loss_weight
