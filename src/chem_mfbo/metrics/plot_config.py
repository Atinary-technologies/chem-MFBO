# plot_config.py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


# plot_config.py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# Matplotlib plotting parameters
def configure_plotting():
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.family': 'sans-serif',
        "font.size": 30,
        "axes.titlesize": 21,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 30,
        'figure.figsize': (12, 12),
        'figure.constrained_layout.use': True,
        'errorbar.capsize': 4.0,
        'lines.markersize': 12,
        'lines.linewidth': 2.0,
        'lines.markeredgewidth': 1.5,
        'legend.borderaxespad': 0.15,
        'legend.borderpad': 0.2,
        'legend.columnspacing': 0.5,
        "legend.handletextpad": 0.5,
        'legend.handlelength': 1.5,
        'legend.handleheight': 0.5,
    })

# Colors for plotting
COLORS_DICT = {
    "mf_ei": cm.plasma(0.1),
    "mf_mes": cm.plasma(0.2),
    "mf_kg": cm.plasma(0.5),
    "sf_kg": cm.plasma(0.6),
    "sf_ei": cm.plasma(0.75),
    "sf_mes": cm.plasma(0.80),
    "random": "k",
}

MARKERS = {
    "mf_mes": "o",
    "sf_mes": "o",
    "sf_ei": "D",
    "mf_ei": "D",
    "random": "x",
}

LINESTYLE = {
    "mf_mes": "--",
    "sf_mes": "-",
    "sf_ei": "-",
    "sf_kg": "-",
    "mf_ei": "--",
    "mf_kg": "--",
    "random": ":",
}

# Problem-specific configurations
PROBLEM_CONFIG = {
    "cofs": {
        "optimum": 18.56,
        "result_col_name": "selectivity",
        "fidelity_name": "fidelity",
    },
    "polarizability": {
        "optimum": 1.0,
        "result_col_name": "Polarizability",
        "fidelity_name": "fidelity",
    },
    "freesolv": {
        "optimum": 25.47,
        "result_col_name": "solvation",
        "fidelity_name": "fidelity",
    },
}


OPTIM = {"branin": -0.397887, "park": 25.5893, "hartmann": 1}
