#env conda biomech_env
from cProfile import label
from pathlib import Path
import ezc3d
import numpy as np
from pyomeca import Analogs, Markers
import matplotlib.pyplot as plt
import biorbd

import numpy as np

from pyomeca import Analogs

MODE_PEDALAGE = "eccentric"
PUISSANCE = "40"


def resample_emg_to_100hz(emg, target_fs=100):
    """
    Robust EMG resampling to target frequency (Hz).
    Handles NaNs and duplicated time stamps.
    """

    # Temps original
    time_old = emg.time.values

    # 1️⃣ Supprimer doublons temporels
    time_unique, unique_idx = np.unique(time_old, return_index=True)
    emg_clean = emg.isel(time=unique_idx)

    # 2️⃣ Supprimer NaN (important après normalize)
    emg_clean = emg_clean.where(np.isfinite(emg_clean), 0)

    # Bornes temporelles strictes
    t_start = time_unique[0]
    t_end = time_unique[-1]

    # 3️⃣ Nouveau vecteur temps STRICT
    dt = 1 / target_fs
    time_new = np.arange(t_start, t_end, dt)

    # 4️⃣ Interpolation sans extrapolation
    emg_resampled = emg_clean.interp(
        time=time_new,
        method="linear",
        kwargs={"fill_value": "extrapolate"}  # ou None si tu préfères
    )

    return emg_resampled


# Lecture du fichier C3D
file = f"/Users/leo/Desktop/Projet/Collecte_25_11/C3D_labelled/{MODE_PEDALAGE}_{PUISSANCE}W.c3d"
file_dir = "/Users/leo/Desktop/Projet/Collecte_25_11/MVC"
print(Analogs.from_c3d(file).name)


trigger_name = ['Electric Resistance.1']

mvc_mapping = {
    "delt_ant": "Sensor 1.IM EMG1",
    "delt_med": "Sensor 2.IM EMG2",
    "delt_post": "Sensor 3.IM EMG3",
    "trap_sup": "Sensor 4.IM EMG4",
    "triceps": "Sensor 5.IM EMG5",
    "biceps": "Sensor 6.IM EMG6",
    "trap_med": "Sensor 7.IM EMG7",
    "trap_inf": "Sensor 8.IM EMG8",
    "gd": "Sensor 9.IM EMG9",
    "pec": "Sensor 10.IM EMG10",
    "brachio": "Sensor 11.IM EMG11",
}

emg_names = list(mvc_mapping.values())
n_emg = len(emg_names)

mvc_blocks = []
fs_analog = None
units = None

for muscle_name, emg_label in mvc_mapping.items():
    mvc_file = Path(file_dir) / f"mvc_{muscle_name}.c3d"

    if not mvc_file.exists():
        print(f"⚠️ Fichier manquant : {mvc_file.name}")
        continue

    print(f"Lecture MVC : {mvc_file.name}")

    # Lire uniquement le canal EMG voulu
    mvc_trial = Analogs.from_c3d(
        mvc_file,
        usecols=[emg_label]
    )

    if fs_analog is None:
        fs_analog = mvc_trial.rate
        units = mvc_trial.units

    # Créer un bloc vide (n_emg, n_frames)
    block = np.zeros((n_emg, mvc_trial.values.shape[1]))

    # Position du canal EMG dans la liste finale
    emg_index = emg_names.index(emg_label)

    # Copier le signal MVC au bon endroit
    block[emg_index, :] = mvc_trial.values[0, :]

    mvc_blocks.append(block)

# Concaténation temporelle
mvc_values = np.concatenate(mvc_blocks, axis=1)

# Objet final pyomeca
mvc_raw = Analogs(
    data=mvc_values,
    channels=emg_names
)
mvc_ref = mvc_raw.max(dim="time")


#detection index passage trigger ON
trigger = Analogs.from_c3d(file, usecols=trigger_name).values.squeeze()
emg_raw = Analogs.from_c3d(file, usecols=emg_names)

mvc_files = sorted(Path(file_dir).glob("*.c3d"))

markers_raw = Markers.from_c3d(file, usecols=['Clav_SC'])

fs_analog = emg_raw.rate
trigger_index = np.where(trigger > 4)[0][0]
print("Trigger détecté à l’échantillon :", trigger_index)

# fréquence analogique
trigger_time = trigger_index / fs_analog
trigger_frame = int(trigger_time*markers_raw.rate)

print("markers rate", trigger_frame)

emg = emg_raw.isel(time=slice(trigger_index, None))


emg_processed = (
    emg.meca.band_pass(order=2, cutoff=[10, 425])
    .meca.center()
    .meca.abs()
    .meca.low_pass(order=4, cutoff=5, freq=fs_analog)
    .meca.normalize(mvc_ref) #à normaliser à partir de la MVC
)

emg_processed_resampled = resample_emg_to_100hz(emg_processed)/100 #entre 0 et 1
emg_resampled = resample_emg_to_100hz(emg)

n_channels = emg_processed_resampled.shape[0]
n_cols = 3
n_rows = int(np.ceil(n_channels / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

muscle_names = list(mvc_mapping.keys())  # ordre déjà correct

for i in range(n_channels):
    ax = axes[i]
    frames_raw = np.arange(emg_resampled.values.shape[1])
    frames_proc = np.arange(emg_processed_resampled.values.shape[1])
    ax.plot(frames_raw, emg_resampled.values[i]*1000, label="Raw")
    ax.plot(frames_proc, emg_processed_resampled.values[i], label="Processed")
    ax.set_title(muscle_names[i])
    ax.legend()
    ax.grid(True)

# Désactiver les axes en trop
for j in range(n_channels, len(axes)):
    axes[j].axis("off")

plt.tight_layout()

np.save(
    f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/emg_processed_resampled.npy",
    emg_processed_resampled.values
)

plt.show()
