import numpy as np
import biorbd
import matplotlib.pyplot as plt
import ezc3d
from pyomeca import Analogs


model_path = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod"
c3d_path   = "/Users/leo/Desktop/Projet/Collecte_25_11/C3D_labelled/concentric_40W.c3d"
q_path     = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/inverse_kinematic_sidonie_40W.npy"

def find_trigger(file):
    # Charger le canal analogique
    analog = Analogs.from_c3d(filename=file, usecols=['Electric Resistance.1']).values[0]

    # Charger le c3d
    c3d = ezc3d.c3d(file)

    # Lire les fréquences
    analog_rate = c3d["parameters"]["ANALOG"]["RATE"]["value"][0]  # ex: 2000
    point_rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]  # ex: 100

    # ratio entre analog et markers
    ratio = int(analog_rate / point_rate)  # ex: 20

    # Trouver les indices (en samples analogiques) où le signal dépasse 2V
    trigger_samples = np.where(analog > 2.0)[0]

    # Si rien ne dépasse → on renvoie 0
    if trigger_samples.size == 0:
        return 0

    # Premier sample dépassant 2V
    first_trigger_sample = trigger_samples[0]

    # Convertir en frame markers
    trigger_frame = first_trigger_sample // ratio

    print("Trigger (sample analog) =", first_trigger_sample)
    print("Trigger (frame marker)  =", trigger_frame)

    return trigger_frame

model = biorbd.Model(model_path)
q_recons = np.load(q_path)                      # (nQ, n_frames)
n_frames = q_recons.shape[1]
print("Q shape :", q_recons.shape)

c3d = ezc3d.c3d(c3d_path)
markers_exp = c3d["data"]["points"][:3, :,find_trigger(str(c3d_path)):]  # shape = (3, n_markers_c3d, n_frames)
markers_exp = markers_exp / 1000
c3d_marker_names = c3d["parameters"]["POINT"]["LABELS"]["value"]
n_markers_c3d = markers_exp.shape[1]

print("Markers C3D :", c3d_marker_names)

model_marker_names = [m.to_string() for m in model.markerNames()]
n_markers_model = len(model_marker_names)
print("Markers modèle :", model_marker_names)

mapping = []
for i_model, name in enumerate(model_marker_names):
    if name in c3d_marker_names:
        i_c3d = c3d_marker_names.index(name)
        mapping.append((i_model, i_c3d))

n_common = len(mapping)
print("Mapping :", mapping)
print("Nombre de markers communs :", n_common)

marker_error = np.zeros((n_frames, n_common))

for f in range(n_frames):

    # Get pose Q at frame f
    Qf = biorbd.GeneralizedCoordinates(q_recons[:, f])

    # Model marker positions
    markers_model = model.markers(Qf)
    markers_model = np.array([m.to_array() for m in markers_model]).T  # (3, n_model_markers)

    errors_f = np.zeros(n_common)

    for k, (i_model, i_c3d) in enumerate(mapping):

        m_model = markers_model[:, i_model]
        m_exp   = markers_exp[:, i_c3d, f]

        if np.isnan(m_exp).any():
            errors_f[k] = 0.0  # marker absent → erreur = 0
        else:
            errors_f[k] = np.linalg.norm(m_model - m_exp)

    marker_error[f, :] = errors_f

# mm conversion
marker_error_mm = marker_error * 1000

plt.figure(figsize=(16, 8))

for k, (i_model, i_c3d) in enumerate(mapping):
    name = model_marker_names[i_model]
    plt.plot(marker_error_mm[:, k], label=name)

#plt.plot(np.mean(marker_error_mm, axis=1), 'k--', linewidth=3, label="Erreur moyenne")

plt.title("Erreur de reconstruction IK par marker")
plt.xlabel("Frame")
plt.ylabel("Erreur (mm)")
plt.legend(fontsize=7)
plt.grid(True)
plt.tight_layout()
plt.show()
