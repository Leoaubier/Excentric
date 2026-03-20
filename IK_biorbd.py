from pathlib import Path
from math import pi
from pyomeca import Analogs
import numpy as np
import matplotlib.pyplot as plt
import ezc3d
import biorbd
from scipy.signal import find_peaks
try:
    import bioviz

    biorbd_viz_found = True
except ModuleNotFoundError:
    biorbd_viz_found = False

ESSAI = "Collecte_18_03"
MODE_PEDALAGE = "concentric"
PUISSANCE = "40"


# Choix des frames à analyser
END_FRAME   = None    # Dernière frame (None = dernière frame du fichier)

# 1. Markers du modèle, DANS L'ORDRE DU .bioMod
MODEL_MARKERS = [
    "Ster",
    #"Xiph",
    "C7",
    "T10",
    "Clav_SC",
    "Clav_AC",
    "Clav_Mid",
    "Scap_AA",
    "Scap_TS",
    "Scap_IA",
    "Delt",
    "EpicI",
    "EpicM",
    "ArmI",
    "Elbow",
    "StylU",
    "LArmI",
    "StylR",
    "Index_Base",
    "Little_Base",
    "Hand_Top",
]

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

def build_marker_mapping(c3d_labels):
    mapping = {}
    for name in MODEL_MARKERS:
        if name not in c3d_labels:
            raise ValueError(f"Le marqueur {name} du modèle est absent du C3D.")
        mapping[name] = c3d_labels.index(name)
    return mapping


def extract_relevant_markers(raw_markers, mapping):
    indices = [mapping[name] for name in MODEL_MARKERS]
    return raw_markers[:, indices, :]


def extract_cycles(signal_deg, peaks):
    cycles = []
    for i in range(len(peaks) - 1):
        cyc = signal_deg[peaks[i]:peaks[i + 1]]
        cyc_norm = np.interp(
            np.linspace(0, 1, 200),
            np.linspace(0, 1, len(cyc)),
            cyc
        )
        cycles.append(cyc_norm)
    return np.array(cycles)

def plot_cycles_from_layout(
    signal,             # array (n_dof, n_frames)
    dof_name,           # liste des noms de DoF (len = n_dof)
    layout,             # dict {segment: [(dof_full_name, title), ...]}
    ref_dof_name,       # DoF utilisé pour détecter les cycles
    first_frame,
    end_frame,
    n_points=200,
    ylabel="Value",
    distance_peaks=100,
    labels=None
):
    """
    Trace des cycles normalisés (moyenne ± std + cycles individuels)
    organisés selon un layout multi-lignes (segments).

    signal        : ndarray (n_dof, n_frames)
    dof_name      : list[str]
    layout        : dict segment -> [(dof_full_name, title), ...]
    ref_dof_name  : nom du DoF servant à détecter les cycles
    first_frame   : int
    end_frame     : int or None
    """

    assert ref_dof_name in dof_name, f"{ref_dof_name} absent de dof_name"


    # Sélection temporelle
    sig_sel = signal[:, first_frame:end_frame]

    # Détection des cycles
    ref_idx = dof_name.index(ref_dof_name)
    ref_signal = sig_sel[ref_idx, :]

    peaks, _ = find_peaks(ref_signal, distance=distance_peaks)
    print("Nombre de cycles détectés :", len(peaks) - 1)


    plt.figure(figsize=(12, 3))
    plt.plot(ref_signal, label="q[14,:]")
    plt.plot(peaks, ref_signal[peaks], "ro", label="peaks")
    plt.legend()
    plt.title(f"q[14,:] + peaks détectés (N={len(peaks)})")
    plt.show()


    # Extraction cycles pour chaque DoF du layout
    cycles_per_dof = {}
    mean_per_dof   = {}
    std_per_dof    = {}

    for seg, items in layout.items():
        for dof_full, _title in items:

            if dof_full not in dof_name:
                print(f"[WARN] DoF absent du modèle : {dof_full}")
                continue

            idx = dof_name.index(dof_full)
            sig = sig_sel[idx, :]

            cyc = extract_cycles(sig, peaks)   # (n_cycles, n_points)

            cycles_per_dof[dof_full] = cyc
            mean_per_dof[dof_full]   = np.mean(cyc, axis=0)
            std_per_dof[dof_full]    = np.std(cyc, axis=0)

    # Plot grille
    x = np.linspace(0, 100, n_points)

    segments = list(layout.keys())
    n_rows = len(segments)
    n_cols = max(len(layout[s]) for s in segments)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(11.69, 8.27),
        sharex=True
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.93,
        bottom=0.08,
        wspace=0.25,
        hspace=0.35
    )

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, None]

    for r, seg in enumerate(segments):
        row_items = layout[seg]

        for c in range(n_cols):
            ax = axes[r, c]

            if c >= len(row_items):
                ax.axis("off")
                continue

            dof_full, title = row_items[c]

            if dof_full not in cycles_per_dof:
                ax.set_title(f"{title}\n(MISSING)", fontsize=12)
                ax.axis("off")
                continue

            cycles = cycles_per_dof[dof_full]
            mean_  = mean_per_dof[dof_full]
            std_   = std_per_dof[dof_full]

            # cycles individuels
            #for cc in cycles:
            #    ax.plot(x, cc, color="gray", alpha=0.25, linewidth=1)

            # moyenne + std
            ax.plot(x, mean_, linewidth=2)
            ax.fill_between(x, mean_ - std_, mean_ + std_, alpha=0.2)

            ax.set_title(title, fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)

            if c == 0:
                ax.set_ylabel(f"{seg}\n{ylabel}")
            else:
                ax.set_ylabel("")

            if r == n_rows - 1:
                ax.set_xlabel("Mean cycle (%)")

    # Une seule légende globale (en haut à droite, comme ton exemple)
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(handles, leg_labels, loc="upper right", frameon=False)

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # laisse de la place à la légende
    plt.show()





def main(show=True):

    model_path = Path(f"/Users/leo/Desktop/Projet/{ESSAI}/model_{ESSAI}.bioMod")
    c3d_path = Path(f"/Users/leo/Desktop/Projet/{ESSAI}/C3D_labelled/{MODE_PEDALAGE}_{PUISSANCE}W.c3d")

    model = biorbd.Biorbd(str(model_path))
    nq = model.nb_q

    c3d = ezc3d.c3d(str(c3d_path))
    raw_markers = c3d["data"]["points"][:3, :, :]
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    units = c3d["parameters"]["POINT"]["UNITS"]["value"][0]

    if units.lower().startswith("mm"):
        raw_markers /= 1000.0

    mapping = build_marker_mapping(labels)
    markers = extract_relevant_markers(raw_markers, mapping)

    # === APPLY FRAME SELECTION HERE ===
    markers = markers[:, :, find_trigger(str(c3d_path)):END_FRAME] #find_trigger(str(c3d_path))
    n_frames = markers.shape[2]

    markers = markers.transpose(2, 0, 1)  # => (n_frames, 3, nbDoF)

    markers = [frame for frame in markers]  # => liste de matrices

    q_recons = np.zeros((nq, n_frames))
    qdot_recons = np.zeros((nq, n_frames))
    qddot_recons = np.zeros((nq, n_frames))

    if ESSAI == "Collecte_25_11":
        if MODE_PEDALAGE == "concentric": #vérifier les frames d'initialisations
            if PUISSANCE == "40":
                init = 4000
                dephasage = 0
            elif PUISSANCE == "60":
                init = 3000
                dephasage = 0
            elif PUISSANCE == "80":
                init = 3000
                dephasage = 0
            else:
                print("PB PUISSANCE")
        elif MODE_PEDALAGE == "eccentric":
            if PUISSANCE == "40":
                init = 1000
                dephasage = 0
            elif PUISSANCE == "60":
                init = 3000
                dephasage = 0
            elif PUISSANCE == "80":
                init = 8000
                dephasage = 0
            else:
                print("PB PUISSANCE")
        else:
            print("PB MODE PEDALAGE")
        print("Filtre de Kalman initialisé à la frame", init)

    elif ESSAI == "Collecte_13_03":
        init = 2400
        dephasage = 0

    elif ESSAI == "Collecte_18_03":
        if MODE_PEDALAGE == "concentric": #vérifier les frames d'initialisations
            if PUISSANCE == "40":
                init = 2000
                dephasage = 0
            elif PUISSANCE == "60":
                init = 3000
                dephasage = 0
            elif PUISSANCE == "80":
                init = 3000
                dephasage = 0
            else:
                print("PB PUISSANCE")
        elif MODE_PEDALAGE == "eccentric":
            if PUISSANCE == "40":
                init = 5000
                dephasage = 0
            elif PUISSANCE == "60":
                init = 3000
                dephasage = 0
            elif PUISSANCE == "80":
                init = 8000
                dephasage = 0
            else:
                print("PB PUISSANCE")
        else:
            print("PB MODE PEDALAGE")
        print("Filtre de Kalman initialisé à la frame", init)

    kalman = biorbd.ExtendedKalmanFilterMarkers(model, frequency=100)
    q_i, _, _ = kalman.reconstruct_frame(markers[init])
    for i, (q_i, qdot_i, qddot_i) in enumerate(kalman.reconstruct_frames(markers)):
        q_recons[:, i] = q_i
        qdot_recons[:, i] = qdot_i
        qddot_recons[:, i] = qddot_i

        if i % 200 == 0:
            print(f"Frame {i}/{n_frames}")

    print("IK Kalmann terminé.")
    q_recons = q_recons[:,dephasage:]
    qdot_recons = qdot_recons[:,dephasage:]
    qddot_recons = qddot_recons[:,dephasage:]

    q_cont = q_recons.copy()

    if MODE_PEDALAGE == "concentric": #vérifier les frames d'initialisations
        if PUISSANCE == "40":
            q_cont[8, :] = (q_cont[8, :] + 2 * pi)  # plot du concentric 40W
            q_cont[11, :] = (q_cont[11, :] - pi) % (2 * pi)
            q_cont[12, :] = (-q_cont[12, :]) % (2 * pi)
            q_cont[13, :] = (q_cont[13, :] - pi)
        elif PUISSANCE == "60":
            pass
            #q_recons[8, :] = (q_recons[8, :] + 2*pi) #plot du concentric 40W

        elif PUISSANCE == "80":
            pass
            #q_recons[8, :] = (q_recons[8, :] + 2*pi) #plot du concentric 40W

        else:
            print("PB PUISSANCE")
    elif MODE_PEDALAGE == "eccentric":
        if PUISSANCE == "40":
            q_cont[8, :] = (q_cont[8, :] + 2 * pi)  # plot du concentric 40W
            q_cont[10, :] = (q_cont[10, :] + 2* pi)
            q_cont[11, :] = (q_cont[11, :] + pi)
            q_cont[12, :] = (-q_cont[12, :])
            q_cont[13, :] = (q_cont[13, :] - pi)
            q_cont[15, :] = (q_cont[15, :] + pi)
            q_cont[16, :] = -(q_cont[16, :])-pi
            q_cont[17, :] = -(q_cont[17, :])+pi



        elif PUISSANCE == "60":
            pass
            #q_recons[8, :] = (q_recons[8, :] + 2*pi) #plot du concentric 40W

        elif PUISSANCE == "80":
            pass
            #q_recons[8, :] = (q_recons[8, :] + 2*pi) #plot du concentric 40W

        else:
            print("PB PUISSANCE")
    #q_plot = q_recons
    #q_plot[6, :] = (q_plot[6, :] + pi)
    #q_plot[7, :] = (q_plot[7, :] + pi)
    #q_recons[8, :] = (q_recons[8, :] + 2*pi) #plot du concentric 40W
    #q_plot[9, :] = (q_plot[9, :]- pi)
    #q_plot[10, :] = (q_plot[10, :]- pi)
    #q_recons[11, :] = (q_recons[11, :] - pi)%(2*pi)
    #q_recons[12, :] = (-q_recons[12, :])%(2*pi)
    #q_recons[13, :] = (q_recons[13, :] - pi)
    #q_plot[14, :] = (q_plot[14, :])%(2*pi)
    #q_plot[15, :] = (q_plot[15, :])%(2*pi)


    #q_recons[:,:] = np.unwrap(q_recons[:,:])
    JOINTS = {
        "Plan élévation hum": 0,
        "élévation hum": 1,
        "Rot axiale hum": 2
    }

    # 1) Détection des pics via le coude (référence du cycle)
    shoulder_euler = np.rad2deg(q_cont[11:14, :])
    elbow_euler = np.rad2deg(q_cont[14, :])


    plt.plot((np.rad2deg(np.unwrap(q_cont[11, :]))), label="Plan élévation hum kalmann")  #--> Abduction épaule
    plt.plot((np.rad2deg(np.unwrap(q_cont[12, :]))), label="élévation hum kalmann")  #--> Flexion épaule
    plt.plot((np.rad2deg(np.unwrap(q_cont[13, :]))), label="Rot axiale hum kalmann")  #-->
    plt.plot(elbow_euler, label="Coude kalmann")  #--> Flexion coude
    plt.legend()
    plt.show()
    plt.plot(q_cont[16,:], label="add")
    plt.plot(q_cont[17, :], label="flex")
    plt.legend()
    plt.show()

    # ===========================
    # 2) Enregistrement des données
    # ===========================
    np.save(f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/q_inverse_kinematic.npy", q_recons)
    qdot_recons_new = np.gradient(q_recons,1/100, axis=1)
    np.save(f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/qdot_inverse_kinematic.npy", qdot_recons)
    qddot_recons_new = np.gradient(qdot_recons_new, 1/100, axis=1)
    np.save(f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/qddot_inverse_kinematic.npy", qddot_recons)
    print("données IK enregistrées :)")


    q_clean = q_recons.copy()
    qdot_clean = qdot_recons.copy()

    q_clean = np.unwrap(q_clean)
    qdot_clean = np.unwrap(qdot_clean)

    dq = np.diff(q_clean[:,100:4000], axis=1)*100
    qd = qdot_clean[:, 100:4000-1]

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a.dot(b) / den) if den > 0 else np.nan

    c = [corr(dq[i], qd[i]) for i in range(q_recons.shape[0])]
    print(c)  # plus de valeurs négatives attendues
    # 4) Extraction & normalisation des cycles

    FIRST_FRAME_PLOT = 2000
    END_FRAME_PLOT = 5000

    dof_name = list(model.dof_names)

    # angles en degrés
    q_deg = np.rad2deg(np.unwrap(q_cont, axis=1))

    LAYOUT = {
        "Clavicle": [
            ("thorax_offset_sternoclavicular_left_r1_RotX", "Pro/retraction"),
            ("thorax_offset_sternoclavicular_left_r2_RotY", "Depression/Elevation"),
        ],
        "Scapula": [
            ("scapula_left_rotation_transform_RotX", "Pro/retraction"),
            ("scapula_left_rotation_transform_RotY", "Lat/med rotation"),
            ("scapula_left_rotation_transform_RotZ", "Tilt"),
        ],
        "Humerus": [
            ("scapula_left_offset_shoulder_left_plane_RotX", "Plane of elevation"),
            ("scapula_left_offset_shoulder_left_ele_RotY", "Elevation"),
            ("scapula_left_offset_shoulder_left_rotation_RotZ", "Axial rotation"),
        ],
        "Forearm": [
            ("humerus_left_offset_elbow_left_flexion_RotZ", "Flexion/extension"),
            ("ulna_left_offset_pro_sup_left_RotY", "Pronation/supination"),
        ],
    }

    # 3) SUBPLOTS

    plot_cycles_from_layout(
        signal=q_deg,
        dof_name=dof_name,
        layout=LAYOUT,
        ref_dof_name="humerus_left_offset_elbow_left_flexion_RotZ",
        first_frame=FIRST_FRAME_PLOT,
        end_frame=END_FRAME_PLOT,
        ylabel="Joint angle (°)",
        distance_peaks=100,
        labels = [MODE_PEDALAGE + " " + PUISSANCE + " W"]
    )

    plot_cycles_from_layout(
        signal=qdot_recons,
        dof_name=dof_name,
        layout=LAYOUT,
        ref_dof_name="humerus_left_offset_elbow_left_flexion_RotZ",
        first_frame=FIRST_FRAME_PLOT,
        end_frame=END_FRAME_PLOT,
        ylabel="Joint speed (rad.s-1)",
        distance_peaks=100,
        labels=[MODE_PEDALAGE + " " + PUISSANCE + " W"]
    )

    for i in range(qdot_recons.shape[0]):
        print(f"vitesse {model.dof_names[i]}, min {np.min(qdot_recons[i,FIRST_FRAME_PLOT:END_FRAME_PLOT])}, mean {np.mean(qdot_recons[i,FIRST_FRAME_PLOT:END_FRAME_PLOT])}, max {np.max(qdot_recons[i,FIRST_FRAME_PLOT:END_FRAME_PLOT])}")

    # Animate the results if biorbd viz is installed
    if show and biorbd_viz_found:
        modelviz= biorbd.Model(str(model_path))
        b = bioviz.Viz(loaded_model=modelviz, show_local_ref_frame=True)
        b.load_movement(q_recons)
        b.exec()

if __name__ == "__main__":
    main(show=True)
