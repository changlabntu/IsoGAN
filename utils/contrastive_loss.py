from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
def triplet_loss(encoder, anchor, positive, negative, method='flat'):
    loss = nn.TripletMarginLoss(margin=1, p=2)
    featDown = nn.MaxPool2d(kernel_size=16)

    anchor = encoder(anchor, method='encode')[-1]
    positive = encoder(positive, method='encode')[-1]
    negative = encoder(negative, method='encode')[-1]

    anchor = featDown(anchor)
    positive = featDown(positive)
    negative = featDown(negative)

    anchor = anchor.reshape(anchor.shape[0], anchor.shape[1])
    positive = positive.reshape(positive.shape[0], positive.shape[1])
    negative = negative.reshape(negative.shape[0], negative.shape[1])

    if method == 'max':
        pool = nn.MaxPool1d(kernel_size=23)
        anchor = pool(anchor.T).T
        positive = pool(positive.T).T
        negative = pool(negative.T).T
    elif method == 'avg':
        pool = nn.AvgPool1d(kernel_size=23)
        anchor = pool(anchor.T)
        positive = pool(positive.T)
        negative = pool(negative.T)
    elif method == 'flat':
        anchor = torch.flatten(anchor)
        positive = torch.flatten(positive)
        negative = torch.flatten(negative)

    loss_tri = loss(anchor, positive, negative)

    return loss_tri

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
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

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # print(mask.sum(1))

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        if torch.isnan(loss):
            loss = torch.tensor(0.0).cuda()

        return loss


def l2_norm(v):
    fnorm = torch.norm(v, p=2, dim=1, keepdim=True) + 1e-6
    v = v.div(fnorm.expand_as(v))
    return v

class InstanceLoss(nn.Module):
    def __init__(self, gamma = 1):
        super(InstanceLoss, self).__init__()
        self.gamma = gamma

    def forward(self, feature, label = None):
        # Dual-Path Convolutional Image-Text Embeddings with Instance Loss, ACM TOMM 2020
        # https://zdzheng.xyz/files/TOMM20.pdf
        # using cross-entropy loss for every sample if label is not available. else use given label.
        normed_feature = l2_norm(feature)
        sim1 = torch.mm(normed_feature*self.gamma, torch.t(normed_feature))
        #sim2 = sim1.t()
        if label is None:
            sim_label = torch.arange(sim1.size(0)).cuda().detach()
        else:
            _, sim_label = torch.unique(label, return_inverse=True)
        loss = F.cross_entropy(sim1, sim_label) #+ F.cross_entropy(sim2, sim_label)
        return loss


from numpy.testing import assert_almost_equal
class HistogramLoss(torch.nn.Module):
    def __init__(self, num_steps, cuda=True):
        super(HistogramLoss, self).__init__()
        self.step = 2 / (num_steps - 1)
        self.eps = 1 / num_steps
        self.cuda = cuda
        self.t = torch.arange(-1, 1 + self.step, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]
        if self.cuda:
            self.t = self.t.cuda()

    def forward(self, features, classes):
        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            indsa = (s_repeat_floor - (self.t - self.step) > -self.eps) & (
                        s_repeat_floor - (self.t - self.step) < self.eps) & inds
            assert indsa.nonzero().size()[0] == size, ('Another number of bins should be used')
            zeros = torch.zeros((1, indsa.size()[1])).byte()
            if self.cuda:
                zeros = zeros.cuda()
            indsb = torch.cat((indsa, zeros))[1:, :]
            s_repeat_[~(indsb | indsa)] = 0
            # indsa corresponds to the first condition of the second equation of the paper
            s_repeat_[indsa] = (s_repeat_ - self.t + self.step)[indsa] / self.step
            # indsb corresponds to the second condition of the second equation of the paper
            s_repeat_[indsb] = (-s_repeat_ + self.t + self.step)[indsb] / self.step

            return s_repeat_.sum(1) / size

        classes_size = classes.size()[0]
        classes_eq = (classes.repeat(classes_size, 1) == classes.view(-1, 1).repeat(1, classes_size)).data
        dists = torch.mm(features, features.transpose(0, 1))
        assert ((dists > 1 + self.eps).sum().item() + (
                    dists < -1 - self.eps).sum().item()) == 0, 'L2 normalization should be used'
        s_inds = torch.triu(torch.ones(classes_eq.size()), 1).byte()
        if self.cuda:
            s_inds = s_inds.cuda()
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq[s_inds].sum().item()
        neg_size = (~classes_eq[s_inds]).sum().item()
        s = dists[s_inds].view(1, -1)
        s_repeat = s.repeat(self.tsize, 1)
        s_repeat_floor = (torch.floor(s_repeat.data / self.step) * self.step).float()

        histogram_pos = histogram(pos_inds, pos_size)
        assert_almost_equal(histogram_pos.sum().item(), 1, decimal=1,
                            err_msg='Not good positive histogram', verbose=True)
        histogram_neg = histogram(neg_inds, neg_size)
        assert_almost_equal(histogram_neg.sum().item(), 1, decimal=1,
                            err_msg='Not good negative histogram', verbose=True)
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0])
        histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).byte()
        if self.cuda:
            histogram_pos_inds = histogram_pos_inds.cuda()
        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        loss = torch.sum(histogram_neg * histogram_pos_cdf)

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.margin = 0.5

    def forward(self, inputs_col, targets_col, inputs_row, target_row):

        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        epsilon = 1e-5
        loss = list()

        neg_count = list()
        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets_col[i] == target_row)
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            neg_pair_ = torch.masked_select(sim_mat[i], targets_col[i] != target_row)

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            pos_loss = torch.sum(-pos_pair_ + 1)
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
                neg_count.append(len(neg_pair))
            else:
                neg_loss = 0

            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / n  # / all_targets.shape[1]
        return loss

class XbmTripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(XbmTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs_col, targets_col, inputs_row, targets_row):
        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs_col, inputs_row.t())
        # split the positive and negative pairs
        eyes_ = torch.eye(n, dtype=torch.uint8).cuda()
        pos_mask = targets_col.expand(
            targets_row.shape[0], n
        ).t() == targets_row.expand(n, targets_row.shape[0])
        neg_mask = ~pos_mask
        pos_mask[:, :n] = pos_mask[:, :n] ^ (eyes_.bool())

        loss = list()
        neg_count = list()
        for i in range(n):
            pos_pair_idx = torch.nonzero(pos_mask[i, :]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                pos_pair_ = sim_mat[i, pos_pair_idx]
                pos_pair_ = torch.sort(pos_pair_)[0]

                neg_pair_idx = torch.nonzero(neg_mask[i, :]).view(-1)
                neg_pair_ = sim_mat[i, neg_pair_idx]
                neg_pair_ = torch.sort(neg_pair_)[0]

                select_pos_pair_idx = torch.nonzero(
                    pos_pair_ < neg_pair_[-1] + self.margin
                ).view(-1)
                pos_pair = pos_pair_[select_pos_pair_idx]

                select_neg_pair_idx = torch.nonzero(
                    neg_pair_ > max(0.6, pos_pair_[-1]) - self.margin
                ).view(-1)
                neg_pair = neg_pair_[select_neg_pair_idx]

                pos_loss = torch.sum(1 - pos_pair)
                if len(neg_pair) >= 1:
                    neg_loss = torch.sum(neg_pair)
                    neg_count.append(len(neg_pair))
                else:
                    neg_loss = 0
                print(pos_loss, neg_loss)
                loss.append(pos_loss + neg_loss)
            else:
                loss.append(0)
            print(loss)

        loss = sum(loss) / n

        return loss

class MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_POS
        self.scale_neg = cfg.LOSSES.MULTI_SIMILARITY_LOSS.SCALE_NEG
        self.hard_mining = cfg.LOSSES.MULTI_SIMILARITY_LOSS.HARD_MINING

    def forward(self, inputs_col, targets_col, inputs_row, target_row):
        batch_size = inputs_col.size(0)
        sim_mat = torch.matmul(inputs_col, inputs_row.t())

        epsilon = 1e-5
        loss = list()
        neg_count = 0
        for i in range(batch_size):
            pos_pair_ = torch.masked_select(sim_mat[i], target_row == targets_col[i])
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            neg_pair_ = torch.masked_select(sim_mat[i], target_row != targets_col[i])

            # sampling step
            if self.hard_mining:
                neg_pair = neg_pair_[neg_pair_ + self.margin > torch.min(pos_pair_)]
                pos_pair = pos_pair_[pos_pair_ - self.margin < torch.max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
            neg_count += len(neg_pair)

            # weighting step
            pos_loss = (
                1.0
                / self.scale_pos
                * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh)))
                )
            )
            neg_loss = (
                1.0
                / self.scale_neg
                * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh)))
                )
            )
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True).cuda()

        loss = sum(loss) / batch_size
        return loss