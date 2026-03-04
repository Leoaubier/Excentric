import numpy as np
import spm1d
import matplotlib.pyplot as plt
import biorbd
import pandas as pd
from statsmodels.stats.anova import AnovaRM

# ==============================
# Réglages généraux
# ==============================
N_POINTS = 200  # points par cycle pour normalisation
PUISSANCES = [40, 60, 80]
MODES = ["concentric", "eccentric"]

FRAME_RANGES = {
    "concentric_40": (2000, 5200),
    "eccentric_40": (2000, 5000),
    "concentric_60": (2000, 5000),
    "eccentric_60": (1500, 3500),
    "concentric_80": (1500, 4000),
    "eccentric_80": (7000, 10000)
}

MUSCLE_GROUPS = {
    # Elbow
    "Elbow Flexors": ["BIC_brevis", "BIC_long"],
    "Elbow Extensors": ["TRI_lat", "TRI_med", "TRI_long"],

    # Shoulder
    "Shoulder Flexors": ["DeltoideusClavicle_A", "PectoralisMajorClavicle_S", "PectoralisMajorThorax_I", "PectoralisMajorThorax_M"],
    "Shoulder Extensors": ["DeltoideusScapula_P", "LatissimusDorsi_S", "LatissimusDorsi_M", "LatissimusDorsi_I", "TeresMajor"],
    "Shoulder Abductors": ["DeltoideusScapula_M", "DeltoideusClavicle_A", "Supraspinatus_A", "Supraspinatus_P"],
    "Shoulder Adductors": ["PectoralisMinor", "Subscapularis_S", "Subscapularis_M", "Subscapularis_I"],

    # Scapula / Upper Back
    "Scapula Elevators": ["TrapeziusScapula_S", "TrapeziusScapula_M", "Rhomboideus_S", "Rhomboideus_I", "LevatorScapulae"],
    "Scapula Depressors": ["SerratusAnterior_S", "SerratusAnterior_M", "SerratusAnterior_I"],
    "Scapula Protractors": ["SerratusAnterior_S", "SerratusAnterior_M", "SerratusAnterior_I", "PectoralisMinor"],
    "Scapula Retractors": ["TrapeziusScapula_S", "TrapeziusScapula_M", "Rhomboideus_S", "Rhomboideus_I"],

    # Rotator cuff
    "Rotator Cuff - Internal Rotators": ["Subscapularis_S", "Subscapularis_M", "Subscapularis_I", "PectoralisMajorClavicle_S", "PectoralisMajorThorax_I", "PectoralisMajorThorax_M"],
    "Rotator Cuff - External Rotators": ["Infraspinatus_I", "Infraspinatus_S", "TeresMinor", "Supraspinatus_A", "Supraspinatus_P"]
}

# ==============================
# Fonctions utilitaires
# ==============================
def ensure_forward_rotation(crank_angle, *signals):
    crank_angle = np.asarray(crank_angle, float)
    if np.median(np.diff(crank_angle)) < 0:
        crank_angle = crank_angle[::-1]
        signals = [s[..., ::-1] for s in signals]
        print("Rotation inversée → remis dans le sens croissant")
    return (crank_angle, *signals)

def detect_cycles_from_crank(crank_angle, min_cycle_frames=30):
    a = np.asarray(crank_angle)
    da = np.diff(a)
    wraps = np.where(da < -np.pi)[0] + 1
    if len(wraps) == 0:
        return np.array([0])
    valid = [wraps[0]]
    for s in wraps[1:]:
        if s - valid[-1] >= min_cycle_frames:
            valid.append(s)
    return np.array(valid)

def normalize_cycle_by_angle(signal, crank_angle, start, end, n_points):
    theta = crank_angle[start:end]
    y = signal[start:end]
    if theta[1] < theta[0]:
        theta = theta[::-1]
        y = y[::-1]
    theta = np.unwrap(theta)
    theta_uniform = np.linspace(theta[0], theta[-1], n_points)
    y_interp = np.interp(theta_uniform, theta, y)
    return y_interp

def build_cycle_matrix(signal, crank_angle, n_points=N_POINTS, min_cycle_frames=30):
    starts = detect_cycles_from_crank(crank_angle, min_cycle_frames)
    cycles = []
    for i in range(len(starts)-1):
        start = starts[i]
        end = starts[i+1]
        cycle = normalize_cycle_by_angle(signal, crank_angle, start, end, n_points)
        cycles.append(cycle)
    return np.array(cycles)

def build_all_muscle_cycles(act_matrix, crank_angle, n_points=N_POINTS, min_cycle_frames=30):
    n_muscles = act_matrix.shape[0]
    all_cycles = []
    for m in range(n_muscles):
        cycles = build_cycle_matrix(act_matrix[m], crank_angle, n_points, min_cycle_frames)
        all_cycles.append(cycles)
    return np.array(all_cycles)

def remove_defective_cycle_full(data_dict, cycle_idx):
    """
    Supprime un cycle complet dans toutes les données d'un essai
    data_dict : dict avec clés ['q','act','frc','crank','cycles']
    """
    for key in ['act', 'frc', 'cycles']:
        arr = data_dict[key]
        data_dict[key] = np.delete(arr, cycle_idx, axis=1 if key != 'cycles' else 1)
    # Pour le crank et q on supprime les frames correspondantes au cycle interpolé
    # Approximation : suppimer une tranche de N_POINTS
    start_idx = cycle_idx * N_POINTS
    end_idx = (cycle_idx + 1) * N_POINTS
    data_dict['crank'] = np.delete(data_dict['crank'], np.arange(start_idx, end_idx))
    data_dict['q'] = np.delete(data_dict['q'], np.arange(start_idx, end_idx), axis=1)
    return data_dict

# ==============================
# Métriques et analyses
# ==============================
def compute_muscle_metrics(act_cycles, angle_grid):
    n_muscles = act_cycles.shape[0]
    metrics = {}
    for m in range(n_muscles):
        data = act_cycles[m]
        mean_curve = np.mean(data, axis=0)
        metrics[m] = {
            "mean_activation": np.mean(data),
            "max_activation": np.max(data),
            "auc": np.trapz(mean_curve, angle_grid),
            "peak_phase_deg": np.rad2deg(angle_grid[np.argmax(mean_curve)]),
            "std_activation": np.std(data)
        }
    return metrics

def build_metrics_comparison_table(metrics_con, metrics_ecc, muscle_names):
    rows = []
    for m in metrics_con.keys():
        row = {
            "Muscle": muscle_names[m],
            "Mean CON": round(metrics_con[m]["mean_activation"], 3),
            "Mean ECC": round(metrics_ecc[m]["mean_activation"], 3),
            "Max CON": round(metrics_con[m]["max_activation"], 3),
            "Max ECC": round(metrics_ecc[m]["max_activation"], 3),
            "AUC CON": round(metrics_con[m]["auc"], 3),
            "AUC ECC": round(metrics_ecc[m]["auc"], 3),
            "Peak phase CON (deg)": round(metrics_con[m]["peak_phase_deg"], 1),
            "Peak phase ECC (deg)": round(metrics_ecc[m]["peak_phase_deg"], 1),
        }
        row["Δ Mean (%)"] = round(
            100 * (row["Mean ECC"] - row["Mean CON"]) / (row["Mean CON"] + 1e-8), 1
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values("Δ Mean (%)", key=np.abs, ascending=False)
    return df

def compute_group_contribution(act_cycles, muscle_names, groups):
    group_results = {}
    mean_per_muscle = np.mean(act_cycles, axis=(1,2))
    total_activation = np.sum(mean_per_muscle)
    for group_name, muscle_list in groups.items():
        idx = [i for i, name in enumerate(muscle_names)
               if any(muscle in name for muscle in muscle_list)]
        if len(idx) == 0:
            continue
        group_activation = np.sum(mean_per_muscle[idx])
        contribution = 100 * group_activation / (total_activation + 1e-8)
        group_results[group_name] = round(contribution, 2)
    return pd.DataFrame.from_dict(group_results, orient="index", columns=["Contribution %"])

def compute_co_contraction(act_cycles, muscle_idx_A, muscle_idx_B):
    A = np.mean(act_cycles[muscle_idx_A], axis=0)
    B = np.mean(act_cycles[muscle_idx_B], axis=0)
    cci = np.mean((2 * np.minimum(A, B)) / (A + B + 1e-8))
    return cci

def build_cci_table(act_con, act_ecc, muscle_pairs, muscle_names):
    rows = []
    for (m1, m2) in muscle_pairs:
        cci_con = compute_co_contraction(act_con, m1, m2)
        cci_ecc = compute_co_contraction(act_ecc, m1, m2)
        rows.append({
            "Muscle Pair": f"{muscle_names[m1]} / {muscle_names[m2]}",
            "CCI CON": round(cci_con, 3),
            "CCI ECC": round(cci_ecc, 3),
            "Δ CCI": round(cci_ecc - cci_con, 3)
        })
    return pd.DataFrame(rows)

def spm_paired(con, ecc, alpha=0.05):
    con_clean = con.copy()
    ecc_clean = ecc.copy()
    var_con = np.var(con_clean, axis=0)
    var_ecc = np.var(ecc_clean, axis=0)
    zero_var_mask = (var_con == 0) | (var_ecc == 0)
    epsilon = 1e-8
    if np.any(zero_var_mask):
        con_clean[:, zero_var_mask] += epsilon * np.random.randn(con_clean.shape[0], np.sum(zero_var_mask))
        ecc_clean[:, zero_var_mask] += epsilon * np.random.randn(ecc_clean.shape[0], np.sum(zero_var_mask))
    t = spm1d.stats.ttest_paired(con_clean, ecc_clean)
    ti = t.inference(alpha)
    return ti, zero_var_mask

def run_spm_all_muscles(act_con, act_ecc, alpha=0.05):
    results = []
    for m in range(act_con.shape[0]):
        ti, mask = spm_paired(act_con[m], act_ecc[m], alpha)
        results.append((m, ti, mask))
    return results

def plot_spm(ti, angle_grid, mask, muscle_name="muscle"):
    plt.figure(figsize=(10,4))
    ti.plot()
    plt.xlabel("Angle pédalier (deg)")
    plt.ylabel("t-value")
    plt.title(f"SPM : {muscle_name}")
    plt.tight_layout()
    plt.show()

# ==============================
# Chargement multi-essais
# ==============================
def load_all_trials(base_path, model, muscle_names):
    data = {}
    for mode in MODES:
        for power in PUISSANCES:
            key = f"{mode}_{power}"
            start, end = FRAME_RANGES[key]
            q = np.load(f"{base_path}/{mode}_{power}W/q_inverse_kinematic.npy")[:, start:end]
            act = np.load(f"{base_path}/{mode}_{power}W/muscle_activations_nonlinear.npy")[:, :end-start]
            frc = np.load(f"{base_path}/{mode}_{power}W/muscles_forces.npy")[:, :end-start]
            crank = np.load(f"{base_path}/{mode}_{power}W/crank_angle.npy")[start:end]
            if mode=="eccentric":
                crank, q, act, frc = ensure_forward_rotation(crank, q, act, frc)
            cycles = build_all_muscle_cycles(act, crank, N_POINTS)
            data[key] = {"q": q, "act": act, "frc": frc, "crank": crank, "cycles": cycles}
    return data

# ==============================
# Main run : analyse complète
# ==============================
BASE_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11"
model_path = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie_vtp.bioMod"
model = biorbd.Model(model_path)
muscle_names = [model.muscleNames()[i].to_string() for i in range(int(model.nbMuscles()))]

data = load_all_trials(BASE_PATH, model, muscle_names)
angle_grid = np.linspace(0, 2*np.pi, N_POINTS)

# Analyse pour tous les essais
for power in PUISSANCES:
    print(f"\n================== {power} W ==================")
    con_key = f"concentric_{power}"
    ecc_key = f"eccentric_{power}"
    cycles_con = data[con_key]["cycles"]
    cycles_ecc = data[ecc_key]["cycles"]

    # 1️⃣ Métriques globales
    metrics_con = compute_muscle_metrics(cycles_con, angle_grid)
    metrics_ecc = compute_muscle_metrics(cycles_ecc, angle_grid)
    df_metrics = build_metrics_comparison_table(metrics_con, metrics_ecc, muscle_names)
    print("Top 10 muscles par Δ Mean (%) :")
    print(df_metrics.head(10))

    # 2️⃣ Contribution groupes musculaires
    df_group_con = compute_group_contribution(cycles_con, muscle_names, MUSCLE_GROUPS)
    df_group_ecc = compute_group_contribution(cycles_ecc, muscle_names, MUSCLE_GROUPS)
    print("Contribution groupes CON:")
    print(df_group_con)
    print("Contribution groupes ECC:")
    print(df_group_ecc)

    # 3️⃣ Co-contractions
    # Exemple : premières paires de muscles, à adapter
    muscle_pairs = [(0,1), (2,3)]
    df_cci = build_cci_table(cycles_con, cycles_ecc, muscle_pairs, muscle_names)
    print("Co-contraction:")
    print(df_cci)

    # 4️⃣ SPM
    results_spm = run_spm_all_muscles(cycles_con, cycles_ecc)
    # Exemple plot pour muscle 0
    m, ti, mask = results_spm[0]
    plot_spm(ti, angle_grid, mask, muscle_name=muscle_names[m])