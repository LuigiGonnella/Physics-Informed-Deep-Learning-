import torch
import torch.nn.functional as nnF
import torch.nn as nn

#this is the manual alternative to autodiff on NN output 
#autodiff is more accurate but more expensive
#the only problem is that to use autodiff we need the model to explicitly receive (x, t) as input parameters
def burgers_pde_residual(x, t, u):
    # x: (B, Nx, Nt)
    # t: (B, Nx, Nt)
    # u: (B, Nx, Nt)

    nu = 0.01 #viscosity
    delta_x = x[0, 1, 0] - x[0, 0, 0]
    delta_t = t[0, 0, 1] - t[0, 0, 0]

    ut = (u[:, :, 2:] - u[:, :, :-2]) / (2 * delta_t) #simualtion of ut through central differences like we computed for x, but since the time dim has no perdioci condition (like it wasfor x instead) we have to exclude first and last point

    #so from [0, 1, 2, 3, 4]
    # we have [2, 3, 4] - [0, 1, 2]
    # meaning ut(i) = u(i + 1) - u(i - 1) exluding 0 and 4 as `i`


    u_right = torch.roll(u, shifts = -1, dims = 1)
    # if u = [10, 20, 30, 40]
    #then u_right = [20, 30, 40, 10]
    # so u_right[i] is the right element of u[i] according to periodic conditions
    #dims = 1 indicates the rolling on x dimension
    u_left = torch.roll(u, shifts = 1, dims = 1)

    ux_squared = (torch.square(u_right) - torch.square(u_left)) / (4 * delta_x)
    uxx_nu = nu * (u_right + u_left - 2 * u) / (delta_x ** 2)

    ux_squared = ux_squared[:, :, 1:-1] #exclude first and last t to match ut dim
    uxx_nu = uxx_nu[:, :, 1:-1]
    res = ut + ux_squared - uxx_nu

    return res # (B, nx, nt - 2)

def burgers_pde_loss(x, t, u, per_batch = False):

    residual = burgers_pde_residual(x, t, u)
    
    # loss_fn = nn.MSELoss()
    # loss = loss_fn(residual, torch.zeros_like(residual))

    loss = torch.mean(torch.square(residual), dim = [1, 2]) if per_batch else torch.mean(torch.square(residual))

    return loss #(batch_size,) if per_batch else ()


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
