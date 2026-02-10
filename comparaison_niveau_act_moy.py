import numpy as np
import matplotlib.pyplot as plt

import numpy as np

# ---- Noms muscles (référence = ordre des lignes EMG)
muscle_names_all = [
    "delt_ant","delt_med","delt_post","trap_sup","triceps","biceps",
    "trap_med","trap_inf","gd","pec","brachio"
]

EXCLUDE_MUSCLES = [
     "brachio",
    # "pec",
]

# Indices à conserver (CEUX-LÀ serviront partout)
keep_indices = [i for i, name in enumerate(muscle_names_all) if name not in EXCLUDE_MUSCLES]
muscle_names = [muscle_names_all[i] for i in keep_indices]  # noms du plot
n_mus = len(muscle_names)

# ---- Essais
trials = [
    ("concentric", "40", 2000, 6000),
    ("concentric", "60", 2000, 5000),
    ("concentric", "80", 1500, 4000),
    ("eccentric",  "40", 2000, 5000),
    ("eccentric",  "60", 1500, 3500),
    ("eccentric",  "80", 7000, 10000),
]

trial_keys = [f"{mode}_{p}W" for mode, p, _, _ in trials]
n_trials = len(trial_keys)

means = np.zeros((n_trials, n_mus))
stds  = np.zeros((n_trials, n_mus))

for ti, (mode, p, first, last) in enumerate(trials):
    key = f"{mode}_{p}W"
    path = f"/Users/leo/Desktop/Projet/Collecte_25_11/{mode}_{p}W/emg_processed_resampled.npy"
    emg_all = np.load(path)[:, first:last]          # (11, T)

    emg = emg_all[keep_indices, :]                  # (n_mus, T)  <-- FILTRAGE ICI
    means[ti] = np.mean(emg, axis=1)
    stds[ti]  = np.std(emg, axis=1)

sorted_indices = sorted(
    range(len(muscle_names)),
    key=lambda i: (
        0 if muscle_names[i].startswith("delt") else
        1 if muscle_names[i].startswith("trap") else
        2
    )
)

muscle_names = [muscle_names[i] for i in sorted_indices]
means = means[:, sorted_indices]
stds  = stds[:, sorted_indices]



# ------------------------------------------------------------
# PLOT GROUPÉ
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(18, 6))

x = np.arange(n_mus)  # 11 groupes
width = 0.12          # largeur des barres

colors = plt.cm.tab10(np.linspace(0, 1, n_trials))

for i in range(n_trials):
    offset = (i - n_trials/2) * width + width/2
    ax.bar(
        x + offset,
        means[i],
        width,
        yerr=stds[i],
        capsize=3,
        label=trial_keys[i].replace("_", " "),
        color=colors[i]
    )

ax.set_xticks(x)
ax.set_xticklabels(muscle_names, rotation=30, ha="right")
ax.set_ylabel("EMG (moyenne ± std)")
ax.set_title("EMG moyenne ± écart-type par muscle (6 essais)")
ax.grid(True, axis="y", alpha=0.3)

ax.legend(title="Essais", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()
