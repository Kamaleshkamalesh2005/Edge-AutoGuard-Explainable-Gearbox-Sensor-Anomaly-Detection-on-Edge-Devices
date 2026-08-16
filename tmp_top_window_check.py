import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv('PHM09_competition_1/Run_102.csv', header=None)
vals = df.iloc[:, :3].to_numpy()
win = vals[0:1024, :]
normal = vals[1024:2048, :]

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(win[:, i], label='Window 13 Signal %d' % (i + 1), color='tab:red', alpha=0.9)
    ax.plot(normal[:, i], label='Next normal window', color='tab:blue', alpha=0.35)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylabel('Signal %d' % (i + 1))
fig.suptitle('Run_102: top anomalous window (red) vs adjacent normal window (blue)')
plt.tight_layout()
fig.savefig('tmp_top_window_plot.png', dpi=150)
print('saved', Path('tmp_top_window_plot.png').exists())
for i in range(3):
    print('signal_%d_win' % (i+1), {
        'min': float(np.min(win[:, i])),
        'max': float(np.max(win[:, i])),
        'mean': float(np.mean(win[:, i])),
        'std': float(np.std(win[:, i]))
    })
    print('signal_%d_normal' % (i+1), {
        'min': float(np.min(normal[:, i])),
        'max': float(np.max(normal[:, i])),
        'mean': float(np.mean(normal[:, i])),
        'std': float(np.std(normal[:, i]))
    })
