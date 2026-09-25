import torch
from torch.utils.data import Dataset
import numpy as np
from utils import MatReader


class BurgersDataset(Dataset): #one item is one INITIAL CONDITION with its SOLUTION
    def __init__(self, data_path, train=True):
        super().__init__()
        self.train = train
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available() else 'cpu')
        dataloader = MatReader(data_path)
        self.ic_vals = dataloader.read_field("input").to(self.device)  # !Initial conditions, for each sample and for each spatial coordinate I have one initial condition (means at time 0) --> shape is (n_samples, nx)
        self.solutions = dataloader.read_field("output").to(self.device)  # Solution, for each sample, space and time I have a value --> shape (n_samples, nx, nt)
        self.tspan = dataloader.read_field("tspan").to(self.device)

        nx = self.ic_vals.shape[1]
        nt = self.tspan.shape[1]

        # Discretized Grid (Nx, Nt, 2) --> for each spatial-temporal point we associate 2 values --> (x, t), so input[i, j, :] contains [xi, tj]. This is important because `i` is just the indiex, while the spatial value is Xi = i / nx.
        self.grid = torch.from_numpy(
            np.mgrid[0: 1: 1 / nx, 0: 1: 1 / nt]).permute(1, 2, 0).float().to(self.device)

        #mgrid generates 2 grids X and T of shape (nx, nt).
        # Every row i of X will contain values i / nx
        # Every column j of T will contain values equal to j / nt 
        # so the result would be of shape (2, nx, nt) and we reshape to (nx, nt, 2)
        #so every [i, j] cella will contain the value (i / nx, j / nt)

        # Quirk of data: Need to transpose x, t
        self.solutions = self.solutions.permute(0, 2, 1)

    def __len__(self):
        return self.ic_vals.shape[0] #the dataset contains 100 samples (described by the solution `u`) each evaluated on nx spatial points and nt temporal points

    def __getitem__(self, idx): #for each sample. we want [xi, tj, ic]
        #grid is (nx, nt, 2)
        #ic is (nx,) for one sample
        #we should return (nx, nt, 3) concatenating every ic along nx on every nt
        ic = self.ic_vals[idx] #(nx,)
        ic_grid = ic[:, None].expand(-1, self.grid.shape[1]) #None turns (nx,) into (nx, 1) and expand turns (nx, 1) into (nx, nt) duplicating the same value over all columns
        new_sample = torch.cat((self.grid, ic_grid[..., None]), dim=-1) #to cat we need ic_grid to be (nx, nt, 1) with the one being the same ic[nxi] for every columns
        return new_sample, self.solutions[idx] #(nx, nt, 3), (nx, nt)
