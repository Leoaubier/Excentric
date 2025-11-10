from pyomeca import Analogs
import os
import matplotlib.pyplot as plt

prefix = "/mnt/shared/" if os.name == "posix" else "Q:\\"
pref_bis = "/media/mickael/566A33906A336BBD/" if os.name == "posix" else "C:\\"
sep = os.sep

emg_path_signal = "/home/mickael/Documents/Leo/EMG/emg_signal/muscle_activation.c3d"
emg_path_mcv = "/home/mickael/Documents/Leo/EMG/emg_signal/mcv.c3d"

all_analogs = Analogs.from_c3d(emg_path_signal)
print(list(all_analogs.channel.values))

ch = 'Sensor 1.IM EMG1'
emg = Analogs.from_c3d(emg_path_signal, usecols=['Sensor 1.IM EMG1'])

mcv = Analogs.from_c3d(emg_path_mcv, usecols=['Sensor 1.IM EMG1'])

y_emg = emg.sel(channel=ch).values.squeeze()
t_emg = emg.time.values
y_mcv = mcv.sel(channel=ch).values.squeeze()
t_mcv = mcv.time.values

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(t_emg, y_emg, label='EMG', linewidth=1)
ax.plot(t_mcv, y_mcv, label='MCV', linewidth=1)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.grid(True, ls=':')
ax.legend()
plt.tight_layout()
plt.show()



mcv_processed = (
    mcv.meca.band_pass(order=2, cutoff=[10, 425])
    .meca.center()
    .meca.abs()
    .meca.low_pass(order=4, cutoff=5, freq=emg.rate)
)

print('le max de la mcv est', mcv_processed.max())

emg_processed = (
    emg.meca.band_pass(order=2, cutoff=[10, 425])
    .meca.center()
    .meca.abs()
    .meca.low_pass(order=4, cutoff=5, freq=emg.rate)
    .meca.normalize(ref=mcv_processed.max(), scale=100) # on normalise le signal a partir du max de la mcv (apres filtrage)
)

emg_processed.plot(x="time", col="channel")
plt.show()