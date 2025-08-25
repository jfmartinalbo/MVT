from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_stack(v_grid, median, p16, p84, out_png, title="Stacked residual (planet rest)"):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(v_grid, median, lw=1.5)
    ax.fill_between(v_grid, p16, p84, alpha=0.3, linewidth=0)
    ax.axvline(0.0, ls='--')
    ax.set_xlabel('Velocity [km/s]')
    ax.set_ylabel('Residual F/F_OOT - 1')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_nulls_grid(v_grid, stacks: List[tuple], out_png: str):
    fig = plt.figure(figsize=(9, 7))
    titles = [s[0] for s in stacks]
    for i, (_, sr) in enumerate(stacks, start=1):
        ax = fig.add_subplot(2, 2, i)
        ax.plot(sr.v_grid, sr.median, lw=1.2)
        ax.fill_between(sr.v_grid, sr.p16, sr.p84, alpha=0.3, linewidth=0)
        ax.axvline(0.0, ls='--')
        ax.set_xlim(v_grid.min(), v_grid.max())
        ax.set_title(titles[i-1])
        if i in (3,4): ax.set_xlabel('Velocity [km/s]')
        if i in (1,3): ax.set_ylabel('Residual')
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def plot_injection_curves(inj_depths, rec_depths, rec_lo, rec_hi, out_png: str):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.errorbar(np.array(inj_depths)*100.0, np.array(rec_depths)*100.0,
                yerr=[(np.array(rec_depths)-np.array(rec_lo))*100.0,
                      (np.array(rec_hi)-np.array(rec_depths))*100.0],
                fmt='o')
    ax.plot(np.array(inj_depths)*100.0, np.array(inj_depths)*100.0, ls='--')
    ax.set_xlabel('Injected depth [%]')
    ax.set_ylabel('Recovered depth [%]')
    ax.set_title('Injection–Recovery (Na I)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

# Backward-compat wrapper (keeps your previous function name)
def plot_stack_velocity(v_kms, r_stack, lo=None, hi=None, centers_v=None, out_png="fig_7_1_stack.png"):
    plot_stack(v_kms, r_stack, lo if lo is not None else r_stack, hi if hi is not None else r_stack, out_png)