#env conda biomech_env

import ezc3d
import numpy as np
from pyomeca import Analogs, Markers
import matplotlib.pyplot as plt
import biorbd

# Lecture du fichier C3D
file = "/Users/leo/Desktop/Projet/Données/Test_1.c3d"

emg_names = ['Delt_ant.IM EMG1', 'Delt_med.IM EMG2',
      'Delt_post.IM EMG3',
       'Trap_med.IM EMG4', 'Biceps.IM EMG5', 'Triceps.IM EMG6', 'Brachio.IM EMG8']
trigger_name = ['Trigger.1']
markers_names = ['Ster', 'Xiph', 'C7', 'T10', 'Clav_SC', 'Clav_Mid', 'Clav_AC', 'Scap_AA', 'Scap_TS',
       'Scap_IA', 'Delt', 'ArmI', 'EpicI', 'EpicM', 'Elbow', 'LArmI',
       'StylR', 'StylU', 'Hand_Top', 'Little_Base', 'Index_Base']


#detection index passage trigger ON
trigger = Analogs.from_c3d(file, usecols=trigger_name).values.squeeze()
emg_raw = Analogs.from_c3d(file, usecols=emg_names)
markers_raw = Markers.from_c3d(file, usecols=markers_names)

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
    #.meca.normalize() #à normaliser à partir de la MVC
)
print(emg_processed)

emg_processed.plot(x='time', col='channel', col_wrap=3)

markers = markers_raw.isel(time=slice(trigger_frame, None))

StylR= markers.sel(channel="StylR")
StylR.plot(x="time", col="axis", col_wrap=3)
plt.show()