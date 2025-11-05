# mmpose/mmpose/models/losses/ld_loss_ce.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Literal, Optional, Dict, Any
from mmpose.registry import MODELS

@MODELS.register_module()
class LDLossCE(nn.Module):
    """
    Limb-Deficient Loss (论文式(1)的实现, contrastive CE on keypoint pairs).

    作用对象：
      - 输入可以是热图 (B,K,H,W) 或 关键点logit/score (B,K)。
      - 按照 ld_pairs 中定义的互斥对（自然关节, 残端）做 2 分类 CE。
        每组只要能从 GT 判定出“哪一个存在”，就计入该组的 CE；否则跳过。

    关键参数：
      ld_pairs: List[Tuple[int,int]]  # 每个元素=(intact_idx, residual_idx)
      score_from: 'heatmap_max' | 'heatmap_avg' | 'softargmax' | 'logit'
      temperature: float  # softargmax 的温度(越小越“尖锐”)
      presence_rule: 'tw_or_energy' | 'tw_only' | 'energy_only'
        - 如何从 (targets, target_weights) 判定某个通道“存在”
          * tw_or_energy:  (sum(|heatmap|)>eps) OR (tw>eps)   [默认，鲁棒]
          * tw_only:       仅依赖 target_weight
          * energy_only:   仅依赖 heatmap 能量
      ignore_ambiguous: bool  # 当一组里“两路都存在/都不存在”时是否跳过该组
      reduction: 'mean' | 'sum'
      alpha: float            # 与其它损失的权重配合时使用

    输入：
      preds:
        - 若 score_from='logit'：形状 (B,K)
        - 否则：形状 (B,K,H,W)
      targets: (B,K,H,W), 为了 presence 判定；若 score_from='logit' 也需要传入
      target_weights: (B,K,1) 或 (B,K) 或 None
      data_samples: 预留（若未来从样本中读更精确的 ld/presence 掩码，可用）

    返回：
      dict(loss_ld=..., num_pairs=..., num_used_pairs=...)
    """
    def __init__(self,
                 ld_pairs: List[Tuple[int, int]],
                 score_from: Literal['heatmap_max', 'heatmap_avg', 'softargmax', 'logit'] = 'heatmap_max',
                 temperature: float = 0.1,
                 presence_rule: Literal['tw_or_energy', 'tw_only', 'energy_only'] = 'tw_only',
                 reduction: Literal['mean', 'sum'] = 'mean',
                 alpha: float = 1.0,
                 eps: float = 1e-6):
        super().__init__()
        assert reduction in ('mean', 'sum')
        self.ld_pairs = list(ld_pairs)
        self.score_from = score_from
        self.temperature = float(temperature)
        self.presence_rule = presence_rule
        self.reduction = reduction
        self.alpha = float(alpha)
        self.eps = eps

    def _scores_from_heatmap(self, preds: torch.Tensor) -> torch.Tensor:
        # preds: (B,K,H,W) -> scores: (B,K)
        if self.score_from == 'heatmap_max':
            return preds.amax(dim=(2, 3))
        elif self.score_from == 'heatmap_avg':
            return preds.mean(dim=(2, 3))
        elif self.score_from == 'softargmax':
            # 将每通道 HxW 视作分类分布的logit，做 log-sum-exp 聚合得到“置信度”
            B, K, H, W = preds.shape
            x = preds.view(B, K, H * W) / max(self.temperature, 1e-6)
            # 对每通道做 logsumexp，再减去 log(HW) 相当于温度化后的平均对数似然
            return torch.logsumexp(x, dim=-1) - torch.log(torch.tensor(H * W, device=preds.device, dtype=preds.dtype))
        else:
            raise ValueError(f"score_from={self.score_from} 需要 logit 形状输入")

    def _presence_from_gt(self,
                          targets: torch.Tensor,
                          target_weights: Optional[torch.Tensor]) -> torch.Tensor:
        """
        返回 present_mask: (B,K) 的 bool，表示该关键点在该样本中“存在”
        """
        B, K, H, W = targets.shape
        energy = targets.abs().sum(dim=(2, 3))  # (B,K)
        e_present = energy > self.eps

        if target_weights is None:
            tw = torch.zeros_like(energy)
        else:
            if target_weights.dim() == 3 and target_weights.size(-1) == 1:
                tw = target_weights.squeeze(-1)
            else:
                tw = target_weights
            tw = tw.to(energy.dtype)

        if self.presence_rule == 'tw_or_energy':
            present = (e_present | (tw > self.eps))
        elif self.presence_rule == 'tw_only':
            present = (tw > self.eps)
        elif self.presence_rule == 'energy_only':
            present = e_present
        else:
            raise ValueError(self.presence_rule)
        return present

    def forward(self,
                preds: torch.Tensor,
                targets: torch.Tensor,
                target_weights: Optional[torch.Tensor] = None,
                data_samples=None):
        # 1) 取每通道 score/logit
        if self.score_from == 'logit':
            assert preds.dim() == 2, "score_from='logit' 时 preds 应为 (B,K)"
            scores = preds  # (B,K)
        else:
            assert preds.dim() == 4, "score_from!=logit 时 preds 应为 (B,K,H,W)"
            scores = self._scores_from_heatmap(preds)  # (B,K)

        B, K = scores.shape

        # 2) 基于 GT 判定每通道“存在”
        present = self._presence_from_gt(targets, target_weights)  # (B,K), bool

        # 3) 逐组构造 2 类 logits + label
        pair_logits = []
        pair_labels = []
        mask_used = []

        for (idx_intact, idx_residual) in self.ld_pairs:
            s_intact = scores[:, idx_intact]      # (B,)
            s_resid  = scores[:, idx_residual]    # (B,)
            p_intact = present[:, idx_intact]     # (B,)
            p_resid  = present[:, idx_residual]   # (B,)

            # 三种情况：
            #  a) 仅自然关节存在 -> label=0
            #  b) 仅残端存在     -> label=1
            #  c) 二者都存在或都不存在 -> 视为歧义，根据配置选择忽略或保留（这里默认忽略）
            only_intact = (p_intact & (~p_resid))
            only_resid  = ((~p_intact) & p_resid)
            use_mask = only_intact | only_resid  # (B,)
            if(p_intact & p_resid).any():
                raise RuntimeError("Annotation has error")

            if use_mask.any():
                logits_2 = torch.stack([s_intact, s_resid], dim=-1)  # (B,2)
                label = torch.where(only_resid, torch.ones_like(s_intact, dtype=torch.long),
                                             torch.zeros_like(s_intact, dtype=torch.long))  # (B,)
                pair_logits.append(logits_2[use_mask])
                pair_labels.append(label[use_mask])
                mask_used.append(use_mask)

        if len(pair_logits) == 0:
            raise RuntimeError(
                "[LDLossCE] No valid limb pairs in this batch. "
                "Check presence_rule/annotations/ld_pairs/dataloader sampling."
            )

        logits_cat = torch.cat(pair_logits, dim=0)  # (N_eff, 2)
        labels_cat = torch.cat(pair_labels, dim=0)  # (N_eff,)

        ce = F.cross_entropy(logits_cat, labels_cat, reduction=self.reduction)
        loss = self.alpha * ce
        return loss

@MODELS.register_module()
class LDCombinedLoss(nn.Module):
    """
    组合包装器，用于同时计算 HeatmapLoss (例如 MSELoss)
    和 LDPose 的对比式二分类 LD Loss。

    保持 MMPose 的标准签名：
        forward(preds, targets, target_weights, data_samples=None)
    所以无需修改 Head，只需要在配置文件里替换 loss=dict(...)。
    """

    def __init__(self,
                 heatmap_loss: Dict[str, Any],
                 ld_loss: Dict[str, Any],
                 loss_weight: float = 1.0):
        super().__init__()
        # 通过 registry 动态构建内部两个损失
        self.heatmap_loss = MODELS.build(heatmap_loss)
        self.ld_loss = MODELS.build(ld_loss)
        self.loss_weight = float(loss_weight)

    def forward(self,
                preds: torch.Tensor,
                targets: torch.Tensor,
                target_weights: Optional[torch.Tensor] = None,
                data_samples=None):

        # -------- 1) 原始热图损失 (MSE / JS / etc.) --------
        loss_present = self.heatmap_loss(preds, targets, target_weights, data_samples)

        # -------- 2) LDPose 对比式 CE --------
        loss_ld = self.ld_loss(preds, targets, target_weights, data_samples)

        # -------- 3) 加权组合 --------
        total = self.loss_weight * (loss_present + loss_ld)

        return total