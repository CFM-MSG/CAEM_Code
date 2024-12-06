import torch
from torch import nn
import torch.nn.functional as F


class SparseMaxPool(nn.Module):
    def __init__(self, pooling_counts, N):
        super(SparseMaxPool, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3, 2)] + [nn.MaxPool1d(2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x  # fill a diagonal line
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
        return map2d

class SparseAvgPool(nn.Module):
    def __init__(self, pooling_counts, N):
        super(SparseAvgPool, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        poolers = [nn.AvgPool1d(2, 1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.AvgPool1d(3, 2)] + [nn.AvgPool1d(2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x  # fill a diagonal line
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
        return map2d

class SparseConv(nn.Module):
    def __init__(self, pooling_counts, N, hidden_size):
        super(SparseConv, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1
        self.hidden_size = hidden_size
        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        self.convs = nn.ModuleList()
        self.convs.extend([nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(pooling_counts[0])])
        for c in pooling_counts[1:]:
            self.convs.extend(
                [nn.Conv1d(hidden_size, hidden_size, 3, 2)] + [nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x  # fill a diagonal line
        for conv, (i, j) in zip(self.convs, self.maskij):
            x = conv(x)
            map2d[:, :, i, j] = x
        return map2d


def build_feat2d(cfg):
    pooling_counts = cfg.MODEL.MMN.FEAT2D.POOLING_COUNTS  # [15,8,8] anet, [15] charades
    num_clips = cfg.MODEL.MMN.NUM_CLIPS  # 64 anet, 16 charades
    hidden_size = cfg.MODEL.MMN.FEATPOOL.HIDDEN_SIZE  # 512
    if cfg.MODEL.MMN.FEAT2D.NAME == "conv":
        return SparseConv(pooling_counts, num_clips, hidden_size)
    elif cfg.MODEL.MMN.FEAT2D.NAME == "pool":
        return SparseMaxPool(pooling_counts, num_clips)
    else:
        raise NotImplementedError("No such feature 2d method as %s" % cfg.MODEL.MMN.FEAT2D.NAME)

def mask2weight(mask2d, mask_kernel, padding=0):
    # from the feat2d.py,we can know the mask2d is 4-d: B, D, N, N
    weight = F.conv2d(mask2d[None, None, :, :].float(),
                          mask_kernel, padding=padding)[0, 0]
    weight[weight > 0] = 1 / weight[weight > 0]
    return weight


def get_padded_mask_and_weight(mask, conv):
    masked_weight = torch.round(F.conv2d(mask.clone().float(), torch.ones(1, 1, *conv.kernel_size).cuda(), stride=conv.stride, padding=conv.padding, dilation=conv.dilation))
    masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0]  #conv.kernel_size[0] * conv.kernel_size[1]
    padded_mask = masked_weight > 0
    return padded_mask, masked_weight


class ProposalConv(nn.Module):
    # def __init__(self, input_size, hidden_size, k, num_stack_layers, output_size, mask2d, dataset):
    def __init__(self, mask2d, input_size=512, hidden_size=512, k=2, num_stack_layers=8):
        super(ProposalConv, self).__init__()
        self.num_stack_layers = num_stack_layers
        self.mask2d = mask2d[None, None,:,:]
        # Padding to ensure the dimension of the output map2d
        first_padding = (k - 1) * num_stack_layers // 2
        self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size)])
        self.convs = nn.ModuleList(
            [nn.Conv2d(input_size, hidden_size, k, padding=first_padding)]
        )
        for _ in range(num_stack_layers - 1):
            self.convs.append(nn.Conv2d(hidden_size, hidden_size, k))
            self.bn.append(nn.BatchNorm2d(hidden_size))

    def forward(self, x):
        padded_mask = self.mask2d
        for i in range(self.num_stack_layers):
            x = self.bn[i](self.convs[i](x)).relu()
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.convs[i])
            x = x * masked_weight
        clip_feat = x.permute(0,2,3,1)
        return clip_feat