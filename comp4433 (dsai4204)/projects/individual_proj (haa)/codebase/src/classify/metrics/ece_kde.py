# ----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2022 Teodora Popordanoska
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

import torch
from torch import nn
import numpy as np
import pandas as pd


def get_bandwidth(f, device):
    """
    Select a bandwidth for the kernel based on maximizing the leave-one-out likelihood (LOO MLE).

    :param f: The vector containing the probability scores, shape [num_samples, num_classes]
    :param device: The device type: 'cpu' or 'cuda'

    :return: The bandwidth of the kernel
    """
    bandwidths = torch.cat((torch.logspace(start=-5, end=-1, steps=15), torch.linspace(0.2, 1, steps=5)))
    max_b = bandwidths[0]
    max_l = -float('inf')
    n = len(f)
    for b in bandwidths:
        log_kern = get_kernel(f, b, device)
        log_fhat = torch.logsumexp(log_kern, 1) - torch.log(torch.tensor(n-1))
        l = torch.sum(log_fhat)
        if l > max_l:
            max_l = l
            max_b = b

    return max_b


def get_ece_kde(f, y, bandwidth, p, mc_type, device):
    """
    Calculate an estimate of Lp calibration error.

    :param f: The vector containing the probability scores, shape [num_samples, num_classes]
    :param y: The vector containing the labels, shape [num_samples]
    :param bandwidth: The bandwidth of the kernel
    :param p: The p-norm. Typically, p=1 or p=2
    :param mc_type: The type of multiclass calibration: canonical, marginal or top_label
    :param device: The device type: 'cpu' or 'cuda'

    :return: An estimate of Lp calibration error
    """
    check_input(f, bandwidth, mc_type)
    if f.shape[1] == 1:
        return 2 * get_ratio_binary(f, y, bandwidth, p, device)
    else:
        if mc_type == 'canonical':
            return get_ratio_canonical(f, y, bandwidth, p, device)
        elif mc_type == 'marginal':
            return get_ratio_marginal_vect(f, y, bandwidth, p, device)
        elif mc_type == 'top_label':
            return get_ratio_toplabel(f, y, bandwidth, p, device)


def get_ratio_binary(f, y, bandwidth, p, device):
    assert f.shape[1] == 1

    log_kern = get_kernel(f, bandwidth, device)

    return get_kde_for_ece(f, y, log_kern, p)


def get_ratio_canonical(f, y, bandwidth, p, device):
    if f.shape[1] > 60:
        # Slower but more numerically stable implementation for larger number of classes
        return get_ratio_canonical_log(f, y, bandwidth, p, device)

    log_kern = get_kernel(f, bandwidth, device)
    kern = torch.exp(log_kern)

    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    kern_y = torch.matmul(kern, y_onehot)
    den = torch.sum(kern, dim=1)
    # to avoid division by 0
    den = torch.clamp(den, min=1e-10)

    ratio = kern_y / den.unsqueeze(-1)
    ratio = torch.sum(torch.abs(ratio - f)**p, dim=1)

    return torch.mean(ratio)


# Note for training: Make sure there are at least two examples for every class present in the batch, otherwise
# LogsumexpBackward returns nans.
def get_ratio_canonical_log(f, y, bandwidth, p, device='cpu'):
    log_kern = get_kernel(f, bandwidth, device)
    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    log_y = torch.log(y_onehot)
    log_den = torch.logsumexp(log_kern, dim=1)
    final_ratio = 0
    for k in range(f.shape[1]):
        log_kern_y = log_kern + (torch.ones([f.shape[0], 1]) * log_y[:, k].unsqueeze(0))
        log_inner_ratio = torch.logsumexp(log_kern_y, dim=1) - log_den
        inner_ratio = torch.exp(log_inner_ratio)
        inner_diff = torch.abs(inner_ratio - f[:, k])**p
        final_ratio += inner_diff

    return torch.mean(final_ratio)


def get_ratio_marginal_vect(f, y, bandwidth, p, device):
    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    log_kern_vect = beta_kernel(f, f, bandwidth).squeeze()
    log_kern_diag = torch.diag(torch.finfo(torch.float).min * torch.ones(len(f))).to(device)
    # Multiclass case
    log_kern_diag_repeated = f.shape[1] * [log_kern_diag]
    log_kern_diag_repeated = torch.stack(log_kern_diag_repeated, dim=2)
    log_kern_vect = log_kern_vect + log_kern_diag_repeated

    return get_kde_for_ece_vect(f, y_onehot, log_kern_vect, p)


def get_ratio_toplabel(f, y, bandwidth, p, device):
    f_max, indices = torch.max(f, 1)
    f_max = f_max.unsqueeze(-1)
    y_max = (y == indices).to(torch.int)

    return get_ratio_binary(f_max, y_max, bandwidth, p, device)


def get_kde_for_ece_vect(f, y, log_kern, p):
    log_kern_y = log_kern * y
    # Trick: -inf instead of 0 in log space
    log_kern_y[log_kern_y == 0] = torch.finfo(torch.float).min

    log_num = torch.logsumexp(log_kern_y, dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_num - log_den
    ratio = torch.exp(log_ratio)
    ratio = torch.abs(ratio - f)**p

    return torch.sum(torch.mean(ratio, dim=0))


def get_kde_for_ece(f, y, log_kern, p):
    f = f.squeeze()
    N = len(f)
    # Select the entries where y = 1
    idx = torch.where(y == 1)[0]
    if not idx.numel():
        return torch.sum((torch.abs(-f))**p) / N

    if idx.numel() == 1:
        # because of -inf in the vector
        log_kern = torch.cat((log_kern[:idx], log_kern[idx+1:]))
        f_one = f[idx]
        f = torch.cat((f[:idx], f[idx+1:]))

    log_kern_y = torch.index_select(log_kern, 1, idx)

    log_num = torch.logsumexp(log_kern_y, dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_num - log_den
    ratio = torch.exp(log_ratio)
    ratio = torch.abs(ratio - f)**p

    if idx.numel() == 1:
        return (ratio.sum() + f_one ** p)/N

    return torch.mean(ratio)


def get_kernel(f, bandwidth, device):
    # if num_classes == 1
    if f.shape[1] == 1:
        log_kern = beta_kernel(f, f, bandwidth).squeeze()
    else:
        log_kern = dirichlet_kernel(f, bandwidth).squeeze()
    # Trick: -inf on the diagonal
    return log_kern + torch.diag(torch.finfo(torch.float).min * torch.ones(len(f))).to(device)


def beta_kernel(z, zi, bandwidth=0.1):
    p = zi / bandwidth + 1
    q = (1-zi) / bandwidth + 1
    z = z.unsqueeze(-2)

    log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
    log_num = (p-1) * torch.log(z) + (q-1) * torch.log(1-z)
    log_beta_pdf = log_num - log_beta

    return log_beta_pdf


def dirichlet_kernel(z, bandwidth=0.1):
    alphas = z / bandwidth + 1

    log_beta = (torch.sum((torch.lgamma(alphas)), dim=1) - torch.lgamma(torch.sum(alphas, dim=1)))
    log_num = torch.matmul(torch.log(z), (alphas-1).T)
    log_dir_pdf = log_num - log_beta

    return log_dir_pdf


def check_input(f, bandwidth, mc_type):
    assert not isnan(f)
    assert len(f.shape) == 2
    assert bandwidth > 0
    assert torch.min(f) >= 0
    assert torch.max(f) <= 1


def isnan(a):
    return torch.any(torch.isnan(a))


# ----------------------------------------------------------------------------
# Wrappers for easy usage
# ----------------------------------------------------------------------------

def val_to_torch(x, device='cpu'):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.values
    if isinstance(x, (np.ndarray, list)):
        return torch.tensor(x, dtype=torch.float32, device=device)
    return torch.tensor([x], dtype=torch.float32, device=device)


def ece_kde_score(y_true, y_prob, p=1, mc_type='canonical', device='cpu'):
    """
    Sklearn-style wrapper for ECE KDE.
    y_true: array-like of shape (n_samples,)
    y_prob: array-like of shape (n_samples,) or (n_samples, 1) for binary
    """
    y_true = val_to_torch(y_true, device).long()
    y_prob = val_to_torch(y_prob, device)
    
    # Clamp probabilities to avoid 0 * -inf = NaN in Beta kernel
    y_prob = torch.clamp(y_prob, min=1e-6, max=1-1e-6)
    
    if y_prob.ndim == 1:
        y_prob = y_prob.unsqueeze(1)
        
    bandwidth = get_bandwidth(y_prob, device)
    return get_ece_kde(y_prob, y_true, bandwidth, p, mc_type, device).item()


def get_calibration_estimate(y_true, y_prob, device='cpu'):
    """
    Returns (estimated_accuracy, confidence) vectors for plotting.
    Only supports binary classification for now.
    """
    y = val_to_torch(y_true, device).long()
    f = val_to_torch(y_prob, device)
    
    # Clamp probabilities to avoid 0 * -inf = NaN in Beta kernel
    f = torch.clamp(f, min=1e-6, max=1-1e-6)
    
    if f.ndim == 1:
        f = f.unsqueeze(1)
        
    assert f.shape[1] == 1, "Only binary classification supported for calibration curve"
    
    bandwidth = get_bandwidth(f, device)
    log_kern = get_kernel(f, bandwidth, device)
    
    # Reusing logic from get_kde_for_ece but returning the estimated probability
    f_flat = f.squeeze()
    
    # Select entries where y=1 to compute numerator of the posterior P(Y=1|f)
    # Note: The original implementation in get_kde_for_ece does this optimization:
    # "Select the entries where y = 1"
    # This is to compute the numerator: sum_{j: y_j=1} K(f_i, f_j)
    # The denominator is: sum_{j} K(f_i, f_j)
    
    idx = torch.where(y == 1)[0]
    
    # Handle edge cases - if no positives, estimated accuracy is 0
    if not idx.numel():
        return torch.zeros_like(f_flat).cpu().numpy(), f_flat.cpu().numpy()

    # NOTE: The original code does some tricky index manipulation when idx.numel() == 1
    # We will try to follow the standard logic:
    
    log_kern_y = torch.index_select(log_kern, 1, idx) # K(f_i, f_j) where y_j=1
    
    log_num = torch.logsumexp(log_kern_y, dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)
    
    log_ratio = log_num - log_den
    estimated_accuracy = torch.exp(log_ratio)
    
    # If using the original code's logic for single positive sample:
    if idx.numel() == 1:
        # Original code removes the sample itself from calculation?
        # "log_kern = torch.cat((log_kern[:idx], log_kern[idx+1:]))"
        # This seems to be LOO (Leave-One-Out) logic which is used in get_bandwidth but 
        # get_kde_for_ece seems to also have some special handling.
        # However, for plotting the curve, we want the estimate at each point.
        pass

    return estimated_accuracy.detach().cpu().numpy(), f_flat.detach().cpu().numpy()
