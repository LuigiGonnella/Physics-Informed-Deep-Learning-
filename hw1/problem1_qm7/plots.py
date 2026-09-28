import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_rmse(rmses, title, path_dir: Path):

    path_dir.mkdir(parents=True, exist_ok=True)
    rmses = np.array(rmses)

    fig, ax = plt.subplots()

    ax.plot(np.arange(rmses.shape[0]), rmses, color = 'r')
    ax.set_xlabel('step')
    ax.set_ylabel('RMSE (kcal/mol)')
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path_dir / f"{title}.png")

    plt.show()