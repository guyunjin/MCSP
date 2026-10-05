import torch
from torch import nn

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    Supports supervised and SimCLR-style contrastive learning.
    """
    def __init__(self, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.contrast_mode = contrast_mode

    def forward(
        self,
        features,
        labels=None,
        mask=None,
        temperature=0.07,
        device=None,
        sample_weight=None,
    ):
        """Compute supervised contrastive loss with optional sample weights."""
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if device is None:
            device = features.device

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute pairwise contrastive logits.
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature
        )

        # Normalize logits for numerical stability.
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile the positive-pair mask for all views.
        mask = mask.repeat(anchor_count, contrast_count)
        # Exclude self-contrast pairs.
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log probabilities over contrast samples.
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Average log likelihood over positive pairs.
        mask_sum = mask.sum(1)
        mean_log_prob_pos = torch.zeros_like(mask_sum)
        pos_idx = mask_sum > 0
        mean_log_prob_pos[pos_idx] = (mask * log_prob).sum(1)[pos_idx] / mask_sum[pos_idx]

        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        # Apply optional per-sample weighting.
        if sample_weight is not None:
            if not torch.is_tensor(sample_weight):
                sample_weight = torch.tensor(sample_weight, dtype=torch.float32, device=device)
            else:
                sample_weight = sample_weight.to(device=device, dtype=torch.float32)

            if sample_weight.dim() != 1 or sample_weight.numel() != batch_size:
                raise ValueError(
                    f"`sample_weight` must be 1D of shape [bsz], "
                    f"but got shape {tuple(sample_weight.shape)}"
                )

            # Share each sample weight across its views.
            sample_weight = sample_weight.view(1, batch_size).expand(anchor_count, batch_size)
            weighted_loss = loss * sample_weight
            loss = weighted_loss.sum() / (sample_weight.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss
