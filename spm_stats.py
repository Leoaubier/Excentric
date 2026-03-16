import numpy as np
import spm1d
import matplotlib.pyplot as plt
import biorbd
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.gridspec import GridSpec


# ==============================
# Réglages généraux
# ==============================
N_POINTS = 200  # points par cycle pour normalisation
ESSAI = "Collecte_25_11"
PUISSANCES = [40]
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

ANTAGONIST_GROUPS = {

"elbow": (
    ["BIC_brevis", "BIC_long"],
    ["TRI_lat", "TRI_med", "TRI_long"]
),

"shoulder_flex_ext": (
    ["DeltoideusClavicle_A", "PectoralisMajorClavicle_S", "PectoralisMajorThorax_I", "PectoralisMajorThorax_M"],
    ["DeltoideusScapula_P", "LatissimusDorsi_S", "LatissimusDorsi_M", "LatissimusDorsi_I", "TeresMajor"]
),

"scapula_pro_retr": (
    ["SerratusAnterior_S", "SerratusAnterior_M", "SerratusAnterior_I", "PectoralisMinor"],
    ["TrapeziusScapula_S", "TrapeziusScapula_M", "Rhomboideus_S", "Rhomboideus_I"]
),

"scapula_elev_depr": (
    ["TrapeziusScapula_S", "TrapeziusScapula_M", "Rhomboideus_S", "Rhomboideus_I", "LevatorScapulae"],
    ["SerratusAnterior_S", "SerratusAnterior_M", "SerratusAnterior_I"]
)

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
            "auc": np.trapezoid(mean_curve, angle_grid),
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

def get_group_indices(muscle_names, muscle_list):
    idx = [i for i, name in enumerate(muscle_names)
           if any(m in name for m in muscle_list)]
    return idx

def compute_group_co_contraction(act_cycles, idx_a, idx_b):

    # moyenne des muscles dans chaque groupe
    emg_a = np.mean(act_cycles[idx_a], axis=0)   # cycles x angle
    emg_b = np.mean(act_cycles[idx_b], axis=0)

    # moyenne sur les cycles
    emg_a = np.mean(emg_a, axis=0)
    emg_b = np.mean(emg_b, axis=0)

    # CCI point par point
    cci_curve = 2 * np.minimum(emg_a, emg_b) / (emg_a + emg_b + 1e-8)

    # score global
    cci_score = np.mean(cci_curve)

    return cci_score, cci_curve

def compute_all_co_contractions(
        act_con,
        act_ecc,
        muscle_names,
        antagonist_groups):

    rows = []

    for group_name, (group_a, group_b) in antagonist_groups.items():

        idx_a = get_group_indices(muscle_names, group_a)
        idx_b = get_group_indices(muscle_names, group_b)

        if len(idx_a) == 0 or len(idx_b) == 0:
            continue

        cci_con, _ = compute_group_co_contraction(act_con, idx_a, idx_b)
        cci_ecc, _ = compute_group_co_contraction(act_ecc, idx_a, idx_b)

        rows.append({
            "Group": group_name,
            "CCI CON": round(cci_con,3),
            "CCI ECC": round(cci_ecc,3),
            "Δ CCI": round(cci_ecc - cci_con,3)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Δ CCI", key=np.abs, ascending=False)

    return df

def spm_paired(con, ecc, alpha=0.05):
    con_clean = con.copy()
    ecc_clean = ecc.copy()

    var_con = np.var(con_clean, axis=0)
    var_ecc = np.var(ecc_clean, axis=0)

    zero_var_mask = (var_con == 0) | (var_ecc == 0)
    epsilon = 1e-8

    if np.any(zero_var_mask):
        con_clean[:, zero_var_mask] += epsilon * np.random.randn(
            con_clean.shape[0], np.sum(zero_var_mask)
        )
        ecc_clean[:, zero_var_mask] += epsilon * np.random.randn(
            ecc_clean.shape[0], np.sum(zero_var_mask)
        )

    t = spm1d.stats.ttest2(con_clean, ecc_clean)
    ti = t.inference(alpha)

    return ti, zero_var_mask

def run_spm_all_muscles(act_con, act_ecc, muscle_names, alpha=0.05):

    results = []
    n_muscles = act_con.shape[0]
    n_points = act_con.shape[2]

    # matrice pour la heatmap
    t_matrix = np.zeros((n_muscles, n_points))

    for m in range(n_muscles):

        ti, mask = spm_paired(act_con[m], act_ecc[m], alpha)

        # stocker les t-values pour la heatmap
        t_matrix[m, :] = ti.z

        if not ti.h0reject:
            continue

        for cluster in ti.clusters:

            start = int(cluster.endpoints[0])
            end   = int(cluster.endpoints[1])

            if end <= start:
                continue

            con_zone = act_con[m][:, start:end]
            ecc_zone = act_ecc[m][:, start:end]

            if con_zone.size == 0 or ecc_zone.size == 0:
                continue

            mean_con = np.mean(con_zone)
            mean_ecc = np.mean(ecc_zone)

            direction = "CON > ECC" if mean_con > mean_ecc else "ECC > CON"

            mean_t = np.mean(ti.z[start:end])

            start_deg = start / (n_points - 1) * 360
            end_deg   = end   / (n_points - 1) * 360

            results.append({

                "muscle_index": m,
                "muscle_name": muscle_names[m],

                "start_idx": start,
                "end_idx": end,

                "phase_start_deg": round(start_deg,1),
                "phase_end_deg": round(end_deg,1),

                "p_value": float(cluster.p),

                "direction": direction,

                "t_mean": float(mean_t),
                "cluster_size": end - start

            })

    return results, t_matrix

def plot_spm(ti, muscle_name="muscle"):

    angle_grid = np.linspace(0, 100, len(ti.z))

    plt.figure(figsize=(10,4))
    plt.plot(angle_grid, ti.z)
    plt.axhline(ti.zstar, linestyle='--')
    plt.axhline(-ti.zstar, linestyle='--')

    plt.xlabel("Cycle (%)")
    plt.ylabel("t-value")
    plt.title(f"SPM : {muscle_name}")
    plt.tight_layout()
    plt.show()

def compute_cohens_d(con, ecc):
    mean_con = np.mean(con, axis=0)
    mean_ecc = np.mean(ecc, axis=0)
    sd_con = np.std(con, axis=0, ddof=1)
    sd_ecc = np.std(ecc, axis=0, ddof=1)

    pooled_sd = np.sqrt((sd_con**2 + sd_ecc**2) / 2)
    d = (mean_ecc - mean_con) / (pooled_sd + 1e-8)
    return d

def compute_significant_duration(results_spm):
    duration_dict = {}

    for r in results_spm:
        name = r["muscle_name"]
        duration = r["phase_end_deg"] - r["phase_start_deg"]

        if name not in duration_dict:
            duration_dict[name] = 0

        duration_dict[name] += duration

    df = pd.DataFrame.from_dict(duration_dict, orient="index",
                                 columns=["Total Significant %"])
    return df.sort_values("Total Significant %", ascending=False)

def compute_ecc_dominance(results_spm):
    dominance = {}

    for r in results_spm:
        name = r["muscle_name"]
        duration = r["phase_end_deg"] - r["phase_start_deg"]

        if name not in dominance:
            dominance[name] = 0

        if r["direction"] == "ECC > CON":
            dominance[name] += duration
        else:
            dominance[name] -= duration

    df = pd.DataFrame.from_dict(dominance, orient="index",
                                 columns=["ECC Dominance Score (%)"])
    return df.sort_values("ECC Dominance Score (%)", key=np.abs, ascending=False)

def compute_global_reorganization(results_spm):
    total = 0
    for r in results_spm:
        total += r["phase_end_deg"] - r["phase_start_deg"]

    return total

def compute_regional_score(results_spm, groups):
    region_score = {}

    for region, muscle_list in groups.items():
        score = 0
        for r in results_spm:
            if any(m in r["muscle_name"] for m in muscle_list):
                duration = r["phase_end_deg"] - r["phase_start_deg"]
                if r["direction"] == "ECC > CON":
                    score += duration
                else:
                    score -= duration

        region_score[region] = score

    return pd.DataFrame.from_dict(region_score, orient="index",
                                   columns=["Regional ECC Score (%)"])

STABILIZERS = MUSCLE_GROUPS["Rotator Cuff - External Rotators"] + \
              MUSCLE_GROUPS["Rotator Cuff - Internal Rotators"] + \
              MUSCLE_GROUPS["Scapula Retractors"]

GENERATORS = MUSCLE_GROUPS["Shoulder Flexors"] + \
             MUSCLE_GROUPS["Shoulder Extensors"] + \
             MUSCLE_GROUPS["Elbow Extensors"]

def compute_stability_ratio(results_spm):
    stab_score = 0
    gen_score = 0

    for r in results_spm:
        duration = r["phase_end_deg"] - r["phase_start_deg"]

        if any(m in r["muscle_name"] for m in STABILIZERS):
            if r["direction"] == "ECC > CON":
                stab_score += duration

        if any(m in r["muscle_name"] for m in GENERATORS):
            if r["direction"] == "ECC > CON":
                gen_score += duration

    return stab_score / (gen_score + 1e-8)


def plot_spm_heatmap(
        t_values,
        muscle_names,
        muscle_groups,
        n_points=200):

    pedal_angle = np.linspace(0, 360, n_points)

    # ==============================
    # Ordre musculaire par groupes
    # ==============================

    ordered_muscles = []
    group_boundaries = []
    group_labels = []

    for group, muscles in muscle_groups.items():

        idx = [i for i, name in enumerate(muscle_names)
               if any(m in name for m in muscles)]

        if len(idx) == 0:
            continue

        ordered_muscles.extend(idx)

        group_boundaries.append(len(ordered_muscles))
        group_labels.append(group)

    t_values = t_values[ordered_muscles]
    muscle_names = [muscle_names[i] for i in ordered_muscles]

    # ==============================
    # Figure publication
    # ==============================

    fig, ax = plt.subplots(figsize=(14,10))

    vmax = np.percentile(np.abs(t_values), 99)

    im = ax.imshow(
        t_values,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        extent=[0,360,0,len(muscle_names)]
    )

    # ==============================
    # Labels muscles
    # ==============================

    ax.set_yticks(np.arange(len(muscle_names)))
    ax.set_yticklabels(muscle_names, fontsize=9)

    ax.set_xlabel("Pedal angle (°)", fontsize=12)
    ax.set_ylabel("Muscles", fontsize=12)

    ax.set_xticks(np.arange(0,361,60))

    ax.set_title(
        "SPM t-values (ECC vs CON)",
        fontsize=14,
        weight="bold"
    )

    # ==============================
    # Séparation groupes musculaires
    # ==============================

    for b in group_boundaries:
        ax.axhline(b, color="black", linewidth=1)

    # ==============================
    # Colorbar
    # ==============================

    cbar = plt.colorbar(im)
    cbar.set_label("t-value", fontsize=11)

    # ==============================
    # Style publication
    # ==============================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
BASE_PATH = f"/Users/leo/Desktop/Projet/{ESSAI}"
model_path = f"/Users/leo/Desktop/Projet/{ESSAI}/model_{ESSAI}.bioMod"
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

    print(f"N cycle concentric {power}W : {cycles_con.shape[1]}")
    print(f"N cycle eccentric {power}W : {cycles_ecc.shape[1]}")


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

    # 3️⃣ Co-contraction antagonistes

    df_cci = compute_all_co_contractions(
        cycles_con,
        cycles_ecc,
        muscle_names,
        ANTAGONIST_GROUPS
    )

    print("\nCo-contraction par groupe antagoniste :")
    print(df_cci)

    # 4️⃣ SPM

    results_spm, t_matrix = run_spm_all_muscles(
        cycles_con,
        cycles_ecc,
        muscle_names
    )
    print("\nClusters SPM significatifs:")
    for r in results_spm[:10]:
        print(r)

    plot_spm_heatmap(
        t_matrix,
        muscle_names,
        MUSCLE_GROUPS,
        n_points=N_POINTS,
    )

    # Exemple plot pour muscle 0
    r = results_spm[0]
    print(r["muscle_name"])
    #plot_spm(ti, muscle_name=muscle_names[m])

    df_duration = compute_significant_duration(results_spm)
    print("Durée cumulée significative:")
    print(df_duration.head())

    df_dom = compute_ecc_dominance(results_spm)
    print("Dominance ECC:")
    print(df_dom.head())

    df_region = compute_regional_score(results_spm, MUSCLE_GROUPS)
    print("Score régional ECC:")
    print(df_region)

    global_reorg = compute_global_reorganization(results_spm)
    print("Indice global de réorganisation:", global_reorg)

    stability_ratio = compute_stability_ratio(results_spm)
    print("Ratio stabilisateurs / générateurs:", stability_ratio)