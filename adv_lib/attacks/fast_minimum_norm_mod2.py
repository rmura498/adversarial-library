# Adapted from https://github.com/maurapintor/Fast-Minimum-Norm-FMN-Attack

import math
from functools import partial
from typing import Optional

import torch
from torch import nn, Tensor
from torch.autograd import grad
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts

from adv_lib.utils.losses import difference_of_logits
from adv_lib.utils.projections import l1_ball_euclidean_projection

import matplotlib.pyplot as plt
import numpy as np


def l0_projection_(δ: Tensor, ε: Tensor) -> Tensor:
    """In-place l0 projection"""
    δ = δ.flatten(1)
    δ_abs = δ.abs()
    sorted_indices = δ_abs.argsort(dim=1, descending=True).gather(1, (ε.long().unsqueeze(1) - 1).clamp_(min=0))
    thresholds = δ_abs.gather(1, sorted_indices)
    δ.mul_(δ_abs >= thresholds)


def l1_projection_(δ: Tensor, ε: Tensor) -> Tensor:
    """In-place l1 projection"""
    l1_ball_euclidean_projection(x=δ.flatten(1), ε=ε, inplace=True)


def l2_projection_(δ: Tensor, ε: Tensor) -> Tensor:
    """In-place l2 projection"""
    δ = δ.flatten(1)
    l2_norms = δ.norm(p=2, dim=1, keepdim=True).clamp_(min=1e-12)
    δ.mul_(ε.unsqueeze(1) / l2_norms).clamp_(max=1)


def linf_projection_(δ: Tensor, ε: Tensor) -> Tensor:
    """In-place linf projection"""
    δ = δ.flatten(1)
    ε = ε.unsqueeze(1)
    torch.maximum(torch.minimum(δ, ε, out=δ), -ε, out=δ)


def l0_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    n_features = x0[0].numel()
    δ = x1 - x0
    l0_projection_(δ=δ, ε=n_features * ε)
    return δ


def l1_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    threshold = (1 - ε).unsqueeze(1)
    δ = (x1 - x0).flatten(1)
    δ_abs = δ.abs()
    mask = δ_abs > threshold
    mid_points = δ_abs.sub_(threshold).copysign_(δ)
    mid_points.mul_(mask)
    return x0 + mid_points


def l2_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    ε = ε.unsqueeze(1)
    return x0.flatten(1).mul(1 - ε).add_(ε * x1.flatten(1)).view_as(x0)


def linf_mid_points(x0: Tensor, x1: Tensor, ε: Tensor) -> Tensor:
    ε = ε.unsqueeze(1)
    δ = (x1 - x0).flatten(1)
    return x0 + torch.maximum(torch.minimum(δ, ε, out=δ), -ε, out=δ).view_as(x0)


def normalize_01(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def plot_loss_epsilon_over_steps(loss, epsilon, steps, normalize=True):
    if normalize:
        loss = normalize_01(loss)
        epsilon = normalize_01(epsilon)

    fig1, ax1 = plt.subplots()
    ax1.plot(
        torch.arange(0, steps),
        loss,
        label="Loss"
    )
    ax1.plot(
        torch.arange(0, steps),
        epsilon,
        label="Epsilon"
    )
    ax1.grid()
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss/Epsilon")
    fig1.legend()

    plt.show()

def fmn(model: nn.Module,
        inputs: Tensor,
        labels: Tensor,
        norm: float,
        targeted: bool = False,
        steps: int = 10,
        alpha_init: float = 1.0,
        alpha_final: Optional[float] = None,
        gamma_init: float = 0.05,
        gamma_final: float = 0.001,
        starting_points: Optional[Tensor] = None,
        binary_search_steps: int = 10) -> Tensor:
    """
    Fast Minimum-Norm attack from https://arxiv.org/abs/2102.12827.

    Parameters
    ----------
    model : nn.Module
        Model to attack.
    inputs : Tensor
        Inputs to attack. Should be in [0, 1].
    labels : Tensor
        Labels corresponding to the inputs if untargeted, else target labels.
    norm : float
        Norm to minimize in {0, 1, 2 ,float('inf')}.
    targeted : bool
        Whether to perform a targeted attack or not.
    steps : int
        Number of optimization steps.
    alpha_init : float
        Initial step size.
    alpha_final : float
        Final step size after cosine annealing.
    gamma_init : float
        Initial factor by which epsilon is modified: epsilon = epsilon * (1 + or - gamma).
    gamma_final : float
        Final factor, after cosine annealing, by which epsilon is modified.
    starting_points : Tensor
        Optional warm-start for the attack.
    binary_search_steps : int
        Number of binary search steps to find the decision boundary between inputs and starting_points.

    Returns
    -------
    adv_inputs : Tensor
        Modified inputs to be adversarial to the model.

    """
    _dual_projection_mid_points = {
        0: (None, l0_projection_, l0_mid_points),
        1: (float('inf'), l1_projection_, l1_mid_points),
        2: (2, l2_projection_, l2_mid_points),
        float('inf'): (1, linf_projection_, linf_mid_points),
    }
    if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
    device = inputs.device
    batch_size = len(inputs)
    batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
    dual, projection, mid_point = _dual_projection_mid_points[norm]
    alpha_final = alpha_init / 100 if alpha_final is None else alpha_final
    multiplier = 1 if targeted else -1

    # If starting_points is provided, search for the boundary
    if starting_points is not None:
        is_adv = model(starting_points).argmax(dim=1)
        if not is_adv.all():
            raise ValueError('Starting points are not all adversarial.')
        lower_bound = torch.zeros(batch_size, device=device)
        upper_bound = torch.ones(batch_size, device=device)
        for _ in range(binary_search_steps):
            epsilon = (lower_bound + upper_bound) / 2
            mid_points = mid_point(x0=inputs, x1=starting_points, ε=epsilon)
            pred_labels = model(mid_points).argmax(dim=1)
            is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
            lower_bound = torch.where(is_adv, lower_bound, epsilon)
            upper_bound = torch.where(is_adv, epsilon, upper_bound)

        delta = mid_point(x0=inputs, x1=starting_points, ε=epsilon) - inputs
    else:
        delta = torch.zeros_like(inputs)
    delta.requires_grad_(True)

    if norm == 0:
        epsilon = torch.ones(batch_size, device=device) if starting_points is None else delta.flatten(1).norm(p=0, dim=0)
    else:
        epsilon = torch.full((batch_size,), float('inf'), device=device)

    # Init trackers
    worst_norm = torch.maximum(inputs, 1 - inputs).flatten(1).norm(p=norm, dim=1)
    best_norm = worst_norm.clone()
    best_adv = inputs.clone()

    adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

    epsilon.requires_grad_()
    print(f"Initial epsilon value: {epsilon.data.norm(p=norm)}")

    optimizer = torch.optim.SGD
    scheduler = CosineAnnealingLR

    delta_optim = optimizer([delta], lr=alpha_init)
    delta_scheduler = scheduler(delta_optim, T_max=steps, eta_min=alpha_final)

    loss_per_iter = []
    epsilon_per_iter = []

    for i in range(steps):
        delta_optim.zero_grad()

        cosine = (1 + math.cos(math.pi * i / steps)) / 2
        gamma = gamma_final + (gamma_init - gamma_final) * cosine

        delta_norm = delta.data.flatten(1).norm(p=norm, dim=1)
        adv_inputs = inputs + delta
        logits = model(adv_inputs)
        pred_labels = logits.argmax(dim=1)

        if i == 0:
            labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))
            logit_diff_func = partial(difference_of_logits, labels=labels, labels_infhot=labels_infhot)

        logit_diffs = logit_diff_func(logits=logits)
        loss = -(multiplier * logit_diffs)
        # add the loss to the list for printing purposes

        # print(f"epsilon: {torch.linalg.norm(epsilon)}\n")
        # print(f"delta: {delta.shape}\n")

        loss_per_iter.append(loss.sum().clone().detach().numpy())
        epsilon_per_iter.append(torch.linalg.norm(epsilon).clone().detach().numpy())

        loss.sum().backward()

        is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
        is_smaller = delta_norm < best_norm
        is_both = is_adv & is_smaller
        adv_found.logical_or_(is_adv)
        best_norm = torch.where(is_both, delta_norm, best_norm)
        best_adv = torch.where(batch_view(is_both), adv_inputs.detach(), best_adv)

        alpha = delta_scheduler.get_last_lr()[0]
        #print(f"alpha: {alpha} - gamma: {gamma}\n")

        if norm == 0:
            epsilon = torch.where(is_adv,
                            torch.minimum(torch.minimum(epsilon - 1, (epsilon * (1 - gamma)).floor_()), best_norm),
                            torch.maximum(epsilon + 1, (epsilon * (1 + gamma)).floor_()))
            epsilon.clamp_(min=0)
        else:
            distance_to_boundary = loss.detach().abs() / delta.grad.data.flatten(1).norm(p=dual, dim=1).clamp_(min=1e-12)
            epsilon = torch.where(is_adv,
                            torch.minimum(epsilon * (1 - gamma), best_norm),
                            torch.where(adv_found, epsilon * (1 + gamma), delta_norm + distance_to_boundary))
        

        # clip epsilon
        epsilon = torch.minimum(epsilon, worst_norm)
        # normalize gradient
        # normalize gradient
        # gradient ascent step

        delta_optim.step()
        #delta.data.add_(-delta.grad.data, alpha=alpha)
        #delta = -delta.grad.data + alpha

        # project in place
        projection(δ=delta.data, ε=epsilon)

        # clamp
        delta.data.add_(inputs).clamp_(min=0, max=1).sub_(inputs)

        # delta.grad.zero_()
        # scheduler.step()
        delta_scheduler.step()

        # mostrare epsilon al variare delle iterazioni
        # printare la loss
        # printare la distanza

    plot_loss_epsilon_over_steps(
        loss=loss_per_iter,
        epsilon=epsilon_per_iter,
        steps=steps
    )

    return best_adv
