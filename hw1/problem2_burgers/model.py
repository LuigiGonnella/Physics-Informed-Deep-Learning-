import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet2D(nn.Module):

    # You may include additional arguments if you wish.
    def __init__(self, in_channels = 3, hidden_dim = 64, kernel_size = 3, padding = 1, stride = 1, linear_dim = 128):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)

        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.conv3 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fcl1 = nn.Linear(in_features=hidden_dim, out_features=linear_dim) #applies only on last dim preserving previous ones

        self.fcl2 = nn.Linear(in_features=linear_dim, out_features=1) #applies only on last dim preserving previous ones

        
    #we take as input x (B, nx, nt, 3)
    #we want to generate the solution u(x, t) --> (nx, nt)
    def forward(self, x):

        x = x.permute(0, 3, 1, 2) #conv expect channels as first dim --> (B, 3, nx, nt)
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x)) #(B, 64, nx, nt)

        x = x.permute(0, 2, 3, 1) #(B, nx, nt, 64)
        x = F.tanh(self.fcl1(x)) #pointwise transformation only on 64 features, without mixing info across different (x, t)

        x = self.fcl2(x) #(B, nx, nt, 1)

        return x.squeeze(-1) #solution is u(x, t) --> (B, nx, nt)
