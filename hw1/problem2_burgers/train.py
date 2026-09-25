import torch
import torch.nn.functional as nnF
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data import BurgersDataset
from model import ConvNet2D
from pde import burgers_pde_residual, burgers_data_loss
from torch.utils.data import DataLoader

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def train():
    burgers_train = BurgersDataset(
        'data/Burgers_train_1000_visc_0.01.mat', train=True)
    burgers_validation = BurgersDataset(
        'data/Burgers_test_50_visc_0.01.mat', train=False)

    # Hyperparameters
    lr = 5e-3
    batch_size = 16
    epochs = 40



    # Setup optimizer, model, data loader etc.
    model = ConvNet2D()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataloader = DataLoader(burgers_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(burgers_validation, batch_size=batch_size)

    # n_train_batches = len(train_dataloader) #same as math.ceil(len(burgers_train) / batch_size)
    # n_val_batches = len(val_dataloader)

    # Training Loop
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(epochs)):
        print(f"starting epoch {epoch + 1}...")
        tot_train_loss = 0

        model.train()
        for batch in train_dataloader:
            samples, solutions = batch
            samples = samples.to(device)
            solutions = solutions.to(device)

            optimizer.zero_grad()

            predicted = model(samples)

            data_loss = burgers_data_loss(predicted, solutions)

            data_loss.backward()

            optimizer.step()

            batch_loss = float(data_loss.item()) * samples.size(0) #tot loss, no mean
            tot_train_loss += batch_loss

        epoch_loss = float(tot_train_loss / len(burgers_train)) #mean loss over samples
        train_losses.append(epoch_loss)
        print(f"TRAIN epoch {epoch + 1} completed with loss {epoch_loss:2f}.")

        # Validation Loop
        model.eval()
        tot_val_loss = 0

        with torch.no_grad():

            for batch in val_dataloader:
                samples, solutions = batch
                samples = samples.to(device)
                solutions = solutions.to(device)

                predicted = model(samples)

                data_loss = burgers_data_loss(predicted, solutions)

                batch_loss = float(data_loss.item()) * samples.size(0) #tot loss, like no mean
                tot_val_loss += batch_loss

            epoch_loss = float(tot_val_loss / len(burgers_validation)) #mean per sample
            val_losses.append(epoch_loss)
            print(f"VALIDATION epoch {epoch + 1} completed with loss {epoch_loss:2f}.")

    return model, train_losses, val_losses


if __name__ == '__main__':
    torch.manual_seed(0)
    train()
