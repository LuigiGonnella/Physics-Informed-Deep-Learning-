import torch
import torch.nn.functional as nnF


def burgers_pde_residual(x, t, u):
    # x: (B, Nx, Nt)
    # t: (B, Nx, Nt)
    # u: (B, Nx, Nt)

    nu = 0.01

    # TODO
    pass


def burgers_data_loss(predicted, target):
    # Relative L2 Loss
    # Predicted: (B, Nx, Nt)
    # Target: (B, Nx, Nt)
    err = predicted - target
    err_norm = torch.linalg.vector_norm(err)
    target_norm = torch.linalg.vector_norm(target)

    batch_loss = err_norm / target_norm.clamp_min(1e-12) #avoid 0 division, shape (B,)

    loss = batch_loss.mean() #() single value
    return loss
