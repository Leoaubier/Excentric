import numpy as np
import spm1d
import matplotlib.pyplot as plt
import biorbd
import pandas as pd


def create_activation_summary(act_con_cycles, act_ecc_cycles, alpha=0.05, muscle_names=None):
    """
    Crée un tableau synthétique des activations musculaires pour un mémoire.

    Paramètres :
    - act_con_cycles : np.array, shape (n_subjects, n_muscles, n_points)
    - act_ecc_cycles : np.array, shape (n_subjects, n_muscles, n_points)
    - alpha : seuil statistique pour considérer qu'un muscle a une différence notable
    - muscle_names : liste de noms de muscles (optionnel)

    Retourne :
    - pandas.DataFrame avec colonnes : Muscle | Min Con | Max Con | Mean Con | Min Ecc | Max Ecc | Mean Ecc | Diff notable
    """

    n_muscles, n_cycle, n_points = act_con_cycles.shape
    if muscle_names is None:
        muscle_names = [f'Muscle {i}' for i in range(n_muscles)]

    summary_data = []

    for m in range(n_muscles):
        con = act_con_cycles[m, :, :]
        ecc = act_ecc_cycles[m, :, :]

        # Calcul des stats simples
        min_con = np.min(con)
        max_con = np.max(con)
        mean_con = np.mean(con)

        min_ecc = np.min(ecc)
        max_ecc = np.max(ecc)
        mean_ecc = np.mean(ecc)

        # Test SPM pour savoir si différence notable
        try:
            t = spm1d.stats.ttest_paired(con, ecc)
            ti = t.inference(alpha)
            diff_notable = np.any(ti)  # True si SPM détecte différence
        except spm1d.stats._datachecks.SPM1DError:
            diff_notable = False  # ignore muscles avec variance zéro complète

        # On ajoute la ligne uniquement si différence notable
        if diff_notable:
            summary_data.append([
                muscle_names[m],
                round(min_con, 3), round(max_con, 3), round(mean_con, 3),
                round(min_ecc, 3), round(max_ecc, 3), round(mean_ecc, 3),
                "Oui"
            ])

    # Création du tableau pandas
    df_summary = pd.DataFrame(summary_data,
                              columns=["Muscle", "Min Con", "Max Con", "Mean Con",
                                       "Min Ecc", "Max Ecc", "Mean Ecc", "Diff notable"])
    return df_summary


def detect_cycles_from_crank(crank_angle, min_cycle_frames=30):
    """
    Détecte les débuts de cycle à partir des passages par 2*pi, avec un minimum de frames.
    Fonctionne pour sens croissant ou décroissant.
    """
    a = np.asarray(crank_angle, float)
    da = np.diff(a)
    threshold = np.pi

    if np.median(da) > 0:
        wraps = np.where(da < -threshold)[0] + 1
    else:
        wraps = np.where(da > threshold)[0] + 1

    if wraps.size == 0:
        raise RuntimeError("Aucun wrap détecté dans crank_angle.")

    good = [wraps[0]]
    for s in wraps[1:]:
        if s - good[-1] >= min_cycle_frames:
            good.append(s)

    starts = np.array(good, dtype=int)
    if starts.size < 2:
        raise RuntimeError("Pas assez de cycles détectés.")
    return starts

def build_cycle_matrix(signal, crank_angle, n_points=200, min_cycle_frames=30):
    """
    Retourne une matrice (n_cycles, n_points) pour un signal donné,
    en utilisant la nouvelle fonction de détection des cycles.
    """
    starts = detect_cycles_from_crank(crank_angle, min_cycle_frames)
    cycles = []

    for i in range(len(starts)-1):
        start = starts[i]
        end = starts[i+1]

        cycle = normalize_cycle_by_angle(signal, crank_angle, start, end, n_points)
        cycles.append(cycle)

    return np.array(cycles)

def build_all_muscle_cycles(act_matrix, crank_angle, n_points=200, min_cycle_frames=30):
    """
    act_matrix : (n_muscles, n_frames)
    retourne : (n_muscles, n_cycles, n_points)
    """
    n_muscles = act_matrix.shape[0]
    all_cycles = []

    for m in range(n_muscles):
        cycles = build_cycle_matrix(act_matrix[m], crank_angle, n_points, min_cycle_frames)
        all_cycles.append(cycles)

    return np.array(all_cycles)

def normalize_cycle_by_angle(signal, crank_angle, start, end, n_points):
    """
    Interpolation du signal sur grille angulaire uniforme
    """
    theta = crank_angle[start:end]
    y = signal[start:end]

    # rendre angle strictement monotone pour interp
    if theta[1] < theta[0]:
        theta = theta[::-1]
        y = y[::-1]

    theta = np.unwrap(theta)

    theta_uniform = np.linspace(theta[0], theta[-1], n_points)

    y_interp = np.interp(theta_uniform, theta, y)

    return y_interp


def spm_paired(con, ecc, alpha=0.05):
    """
    Effectue un t-test apparié SPM entre con et ecc en ajoutant un petit bruit
    aux time points à variance nulle pour éviter l'erreur SPM.

    con, ecc : np.array de forme (n_sujets, n_timepoints)
    alpha : seuil significatif
    """
    con_clean = con.copy()
    ecc_clean = ecc.copy()

    # Détecter les colonnes (time points) à variance nulle
    var_con = np.var(con_clean, axis=0)
    var_ecc = np.var(ecc_clean, axis=0)
    zero_var_mask = (var_con == 0) | (var_ecc == 0)

    # Ajouter un tout petit bruit epsilon uniquement sur ces colonnes
    epsilon = 1e-8
    if np.any(zero_var_mask):
        con_clean[:, zero_var_mask] += epsilon * np.random.randn(con_clean.shape[0], np.sum(zero_var_mask))
        ecc_clean[:, zero_var_mask] += epsilon * np.random.randn(ecc_clean.shape[0], np.sum(zero_var_mask))

    # Effectuer le t-test apparié
    t = spm1d.stats.ttest_paired(con_clean, ecc_clean)
    ti = t.inference(alpha)

    return ti, zero_var_mask


def run_spm_all_muscles(act_con, act_ecc, alpha=0.05):
    """
    act_con, act_ecc : listes ou arrays shape (n_muscles, n_subjects, n_timepoints)
    Retourne les résultats SPM pour tous les muscles
    """
    n_muscles = len(act_con)
    results = []
    alpha = 0.05

    for m in range(act_con_cycles.shape[0]):  # boucle sur les muscles
        ti, mask = spm_paired(act_con_cycles[m, :, :], act_ecc_cycles[m, :, :], alpha)
        results.append((m, ti, mask))

    # Exemple de lecture :
    for r in results:
        if r is not None:
            m, ti, mask = r
            # plotting ou analyse
    return results


def plot_spm(ti, angle_grid, mask, muscle_name="muscle"):
    if ti is None:
        print(f"{muscle_name} non analysé (variance nulle totale)")
        return

    angle_deg = np.rad2deg(angle_grid)[mask]

    plt.figure(figsize=(10,4))
    ti.plot()
    plt.xlabel("Angle pédalier (deg)")
    plt.ylabel("t-value")
    plt.title(f"SPM : {muscle_name}")
    plt.tight_layout()
    plt.show()
# =====================================================
# CHARGEMENT DONNEES
# =====================================================
PUISSANCE = "40"
FIRST_FRAME_PLOT = 3000
END_FRAME_PLOT = 4000
n_frame = END_FRAME_PLOT - FIRST_FRAME_PLOT

model = biorbd.Model("/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie_vtp.bioMod")
muscle_names = [model.muscleNames()[i].to_string() for i in range(int(model.nbMuscles()))]

# --- Concentrique ---
q_con   = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/q_inverse_kinematic.npy")[:, FIRST_FRAME_PLOT:END_FRAME_PLOT]
act_con = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/muscle_activations_nonlinear.npy")[:, :n_frame-50]
frc_con = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/muscles_forces.npy")[:, :n_frame-50]
crank_con = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/crank_angle.npy")[FIRST_FRAME_PLOT:END_FRAME_PLOT-50]

# --- Excentrique ---
q_ecc   = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/q_inverse_kinematic.npy")[:, FIRST_FRAME_PLOT:END_FRAME_PLOT]
act_ecc = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/muscle_activations_nonlinear.npy")[:, :n_frame]
frc_ecc = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/muscles_forces.npy")[:, :n_frame]
crank_ecc = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/crank_angle.npy")[FIRST_FRAME_PLOT:END_FRAME_PLOT]

if crank_ecc[1] < crank_ecc[0]:
    crank_ecc = crank_ecc[::-1]
    act_ecc = act_ecc[:, ::-1]

# =====================================================
# ANALYSE PAR MUSCLE
# =====================================================

# Nombre de points angulaires
N_POINTS = 200

# Construction cycles
act_con_cycles = build_all_muscle_cycles(act_con, crank_con, N_POINTS)
act_ecc_cycles = build_all_muscle_cycles(act_ecc, crank_ecc, N_POINTS)

print("Shape CON:", act_con_cycles.shape)
print("Shape ECC:", act_ecc_cycles.shape)

# Angle grid commun
angle_grid = np.linspace(0, 2*np.pi, N_POINTS)

# SPM
results = run_spm_all_muscles(act_con_cycles, act_ecc_cycles)

# Exemple muscle 0
m, ti, mask = results[33]
plot_spm(ti, angle_grid, mask, muscle_name="Muscle_30")

df_summary = create_activation_summary(act_con_cycles, act_ecc_cycles, alpha=0.05, muscle_names=muscle_names)

print(df_summary)