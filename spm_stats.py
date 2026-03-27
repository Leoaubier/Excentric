import numpy as np
import spm1d
import matplotlib.pyplot as plt
import biorbd
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.gridspec import GridSpec

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)

# ==============================
# Réglages généraux
# ==============================
N_POINTS = 200  # points par cycle pour normalisation
ESSAI = "Collecte_18_03"
PUISSANCES = [40,60]
MODES = ["concentric", "eccentric"]

if ESSAI == "Collecte_25_11":
    FRAME_RANGES = {
        "concentric_40": (2000, 5200),
        "eccentric_40": (2000, 5000),
        "concentric_60": (2000, 5000),
        "eccentric_60": (1500, 3500),
        "concentric_80": (1500, 4000),
        "eccentric_80": (7000, 10000)
    }
elif ESSAI == "Collecte_18_03":
    FRAME_RANGES = {
        "concentric_40": (2000, 5000),
        "eccentric_40": (5000, 8000),
        "concentric_60": (2000, 5000),
        "eccentric_60": (14000, 17000),
        "concentric_left": (500, 2500),
        "eccentric_left": (4000, 7000)
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
def safe_delta(mean_con, mean_ecc, eps=1e-4):
    if mean_con < eps and mean_ecc < eps:
        return 0.0
    return 100 * (mean_ecc - mean_con) / max(mean_con, eps)

def compute_muscle_metrics(act_cycles, angle_grid):
    n_muscles = act_cycles.shape[0]
    metrics = {}

    angle_deg = np.rad2deg(angle_grid) % 360

    for m in range(n_muscles):
        data = act_cycles[m]  # (n_cycles, n_points)
        mean_curve = np.mean(data, axis=0)

        # ===== AUC =====
        auc = np.trapz(mean_curve, angle_deg)

        # ===== Peak =====
        peak_idx = np.argmax(mean_curve)
        peak_val = mean_curve[peak_idx]
        peak_phase = angle_deg[peak_idx]

        # ===== Centre of Activity (CoA) =====
        if np.sum(mean_curve) > 1e-6:
            coa = np.sum(angle_deg * mean_curve) / np.sum(mean_curve)
        else:
            coa = np.nan

        metrics[m] = {
            "mean_activation": np.mean(data),
            "max_activation": peak_val,
            "auc": auc,
            "peak_phase_deg": peak_phase,
            "CoA": coa,
            "std_activation": np.std(data)
        }

    return metrics

def build_metrics_comparison_table(metrics_con, metrics_ecc, muscle_names):
    rows = []

    for m in metrics_con.keys():

        con = metrics_con[m]
        ecc = metrics_ecc[m]

        row = {
            "Muscle": muscle_names[m],

            "Mean CON": con["mean_activation"],
            "Mean ECC": ecc["mean_activation"],

            "Peak CON": con["max_activation"],
            "Peak ECC": ecc["max_activation"],

            "AUC CON": con["auc"],
            "AUC ECC": ecc["auc"],

            "Peak phase CON": con["peak_phase_deg"],
            "Peak phase ECC": ecc["peak_phase_deg"],

            "CoA CON": con["CoA"],
            "CoA ECC": ecc["CoA"],
        }

        # ===== DELTAS =====
        row["Δ Mean (%)"] = safe_delta(row["Mean CON"], row["Mean ECC"])
        row["Δ Peak"] = row["Peak ECC"] - row["Peak CON"]
        row["Δ AUC"] = row["AUC ECC"] - row["AUC CON"]
        row["Δ Phase"] = row["Peak phase ECC"] - row["Peak phase CON"]
        row["Δ CoA"] = row["CoA ECC"] - row["CoA CON"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # tri intelligent (changement global)
    df = df.sort_values("Δ AUC", key=np.abs, ascending=False)

    return df

def compute_pattern_difference(act_con, act_ecc):
    """
    Différence globale de forme (indépendante du bruit)
    """
    n_muscles = act_con.shape[0]
    pattern_diff = []

    for m in range(n_muscles):
        mean_con = np.mean(act_con[m], axis=0)
        mean_ecc = np.mean(act_ecc[m], axis=0)

        # normalisation (important)
        if np.max(mean_con) > 0:
            mean_con = mean_con / np.max(mean_con)
        if np.max(mean_ecc) > 0:
            mean_ecc = mean_ecc / np.max(mean_ecc)

        diff = np.mean(np.abs(mean_con - mean_ecc))

        pattern_diff.append(diff)

    return np.array(pattern_diff)

def compute_rmse_grouped(con, ecc, muscle_names, groups):
    """
    con, ecc : (n_muscles, n_cycles, n_points)
    """

    results = []

    # ==============================
    # 1️⃣ RMSE globale
    # ==============================
    mean_con = np.nanmean(con, axis=(0,1))
    mean_ecc = np.nanmean(ecc, axis=(0,1))

    valid = ~np.isnan(mean_con) & ~np.isnan(mean_ecc)

    if np.sum(valid) > 10:
        rmse_global = np.sqrt(np.mean((mean_con[valid] - mean_ecc[valid])**2))
    else:
        rmse_global = np.nan

    print("\n===== RMSE Pattern =====")
    print(f"GLOBAL RMSE = {round(rmse_global,4)}")

    # ==============================
    # 2️⃣ RMSE par groupe
    # ==============================
    for group_name, muscle_list in groups.items():

        idx = [i for i, name in enumerate(muscle_names)
               if any(m in name for m in muscle_list)]

        if len(idx) == 0:
            continue

        mean_con_g = np.nanmean(con[idx], axis=(0,1))
        mean_ecc_g = np.nanmean(ecc[idx], axis=(0,1))

        valid = ~np.isnan(mean_con_g) & ~np.isnan(mean_ecc_g)

        if np.sum(valid) < 10:
            rmse = np.nan
        else:
            rmse = np.sqrt(np.mean((mean_con_g[valid] - mean_ecc_g[valid])**2))

        results.append({
            "Group": group_name,
            "RMSE": round(rmse, 4),
            "n_muscles": len(idx)
        })

    df = pd.DataFrame(results)

    # tri du plus différent au plus similaire
    df = df.sort_values("RMSE", ascending=False)

    print("\n--- RMSE par groupe musculaire ---")
    print(df)

    return rmse_global, df

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
    emg_a = np.mean(act_cycles[idx_a], axis=0)  # muscles
    emg_b = np.mean(act_cycles[idx_b], axis=0)

    cci_cycles = 2 * np.minimum(emg_a, emg_b) / (emg_a + emg_b + 1e-8)

    cci_score = np.mean(cci_cycles)
    cci_curve = np.mean(cci_cycles, axis=0)

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

def spm_paired(con, ecc, alpha=0.05, var_threshold=1e-6):
    con_clean = con.copy()
    ecc_clean = ecc.copy()

    # Variance point par point
    var_con = np.var(con_clean, axis=0)
    var_ecc = np.var(ecc_clean, axis=0)

    valid_mask = (var_con > var_threshold) & (var_ecc > var_threshold)

    # Si trop peu de points valides → skip
    if np.sum(valid_mask) < 20:
        return None, valid_mask

    con_valid = con_clean[:, valid_mask]
    ecc_valid = ecc_clean[:, valid_mask]

    # T-test apparié
    t = spm1d.stats.ttest_paired(con_valid, ecc_valid)
    ti = t.inference(alpha)

    return ti, valid_mask

def run_spm_all_muscles(act_con, act_ecc, muscle_names, alpha=0.05, ACTIVATION_THRESHOLD = 0.05):

    results = []
    n_muscles = act_con.shape[0]
    n_points = act_con.shape[2]

    # matrice pour la heatmap
    t_matrix = np.zeros((n_muscles, n_points))

    for m in range(n_muscles):

        ti, mask = spm_paired(act_con[m], act_ecc[m], alpha)

        if ti is None:
            continue
        # stocker les t-values pour la heatmap
        t_full = np.zeros(n_points)
        t_full[:] = np.nan  # ou 0 si tu préfères

        t_full[mask] = ti.z

        t_matrix[m, :] = t_full

        if not ti.h0reject:
            continue

        for cluster in ti.clusters:

            valid_indices = np.where(mask)[0]

            start = valid_indices[int(cluster.endpoints[0])]
            end = valid_indices[int(cluster.endpoints[1])]

            if (end - start) < 5:  # ou 3-10 selon ton sampling
                continue

            con_zone = act_con[m][:, start:end]
            ecc_zone = act_ecc[m][:, start:end]

            if con_zone.size == 0 or ecc_zone.size == 0:
                continue

            mean_con = np.mean(con_zone)
            mean_ecc = np.mean(ecc_zone)

            if max(mean_con, mean_ecc) < ACTIVATION_THRESHOLD: #filtre les zones sans activations
                continue

            if abs(mean_ecc - mean_con) < 0.02:
                continue

            direction = "CON > ECC" if mean_con > mean_ecc else "ECC > CON"

            mean_t = np.mean(ti.z[start:end])

            if np.isnan(mean_t):
                continue

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

        start = r["phase_start_deg"]
        end   = r["phase_end_deg"]

        if np.isnan(start) or np.isnan(end):
            continue

        length = end - start if end >= start else (360 - start + end)

        #  normalisation (important)
        length_norm = length / 360

        t_val = r.get("t_mean", 0)

        if np.isnan(t_val):
            continue

        #  clamp t (évite explosion)
        t_val = np.clip(abs(t_val), 0, 5)

        total += length_norm * t_val

    return total

def compute_pattern_similarity_grouped(con, ecc, muscle_names, groups):
    """
    con, ecc : (n_muscles, n_cycles, n_points)
    """

    results = []

    # ==============================
    # 1️⃣ Corrélation globale
    # ==============================
    mean_con = np.nanmean(con, axis=(0,1))  # moyenne tous muscles + cycles
    mean_ecc = np.nanmean(ecc, axis=(0,1))

    valid = ~np.isnan(mean_con) & ~np.isnan(mean_ecc)

    if np.sum(valid) > 10:
        r_global = np.corrcoef(mean_con[valid], mean_ecc[valid])[0,1]
    else:
        r_global = np.nan

    print("\n===== Pattern similarity =====")
    print(f"GLOBAL r = {round(r_global,3)}")

    # ==============================
    # 2️⃣ Par groupe musculaire
    # ==============================
    for group_name, muscle_list in groups.items():

        idx = [i for i, name in enumerate(muscle_names)
               if any(m in name for m in muscle_list)]

        if len(idx) == 0:
            continue

        # moyenne groupe
        mean_con_g = np.nanmean(con[idx], axis=(0,1))
        mean_ecc_g = np.nanmean(ecc[idx], axis=(0,1))

        valid = ~np.isnan(mean_con_g) & ~np.isnan(mean_ecc_g)

        if np.sum(valid) < 10:
            r = np.nan
        else:
            r = np.corrcoef(mean_con_g[valid], mean_ecc_g[valid])[0,1]

        results.append({
            "Group": group_name,
            "r": round(r,3),
            "n_muscles": len(idx)
        })

    df = pd.DataFrame(results).sort_values("r")

    print("\n--- Par groupe musculaire ---")
    print(df)

    return r_global, df

def compute_pattern_rmse(con, ecc, idx):
    mean_con = np.nanmean(con[idx], axis=(0,1))
    mean_ecc = np.nanmean(ecc[idx], axis=(0,1))

    valid = ~np.isnan(mean_con) & ~np.isnan(mean_ecc)

    return np.sqrt(np.mean((mean_con[valid] - mean_ecc[valid])**2))

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
            stab_score += duration if r["direction"] == "ECC > CON" else -duration

        if any(m in r["muscle_name"] for m in GENERATORS):
            gen_score += duration if r["direction"] == "ECC > CON" else -duration

    return stab_score / (gen_score + 1e-8)

def format_group_label(label, max_len=12):
    words = label.split()
    lines = []
    current_line = ""

    for w in words:
        if len(current_line) + len(w) + 1 <= max_len:
            current_line += (" " + w if current_line else w)
        else:
            lines.append(current_line)
            current_line = w

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)

import numpy as np


def plot_spm_heatmap(
        results_spm,
        muscle_names,
        muscle_groups,
        t_matrix,
        n_points=200):

    # ==============================
    # 1. Construire ordre + groupes FIXES
    # ==============================

    ordered_muscles = []
    group_info = []  # (start, end, group_name)

    current_idx = 0

    for group, muscles in muscle_groups.items():

        idx = [i for i, name in enumerate(muscle_names)
               if any(m in name for m in muscles)]

        if len(idx) == 0:
            continue

        ordered_muscles.extend(idx)

        start = current_idx
        end = current_idx + len(idx)

        group_info.append((start, end, group))

        current_idx = end

    # Reorder UNE seule fois
    muscle_names_ord = [muscle_names[i] for i in ordered_muscles]

    # Mapping rapide (évite index())
    muscle_map = {old: new for new, old in enumerate(ordered_muscles)}

    # ==============================
    # 2. Matrice significativité
    # ==============================

    n_muscles = len(muscle_names_ord)
    sig_matrix = np.zeros((n_muscles, n_points))

    for r in results_spm:

        m = r["muscle_index"]

        if m not in muscle_map:
            continue

        new_idx = muscle_map[m]

        start = r["start_idx"]
        end = r["end_idx"]

        if end <= start:
            continue

        valid_zone = ~np.isnan(t_matrix[m, start:end])

        if np.sum(valid_zone) == 0:
            continue

        indices = np.arange(start, end)[valid_zone]

        if r["direction"] == "ECC > CON":
            sig_matrix[new_idx, indices] = 1
        else:
            sig_matrix[new_idx, indices] = -1

    # ==============================
    # Masquer les colonnes invalides (aucun test SPM)
    # ==============================

    invalid_cols = np.all(np.isnan(t_matrix), axis=0)
    sig_matrix[:, invalid_cols] = np.nan

    # ==============================
    # 3. Plot
    # ==============================

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(1, 2, width_ratios=[1, 5], wspace=0.05)

    ax_group = fig.add_subplot(gs[0, 0])  # colonne groupes
    ax = fig.add_subplot(gs[0, 1])  # heatmap

    sig_matrix[:, np.all(np.isnan(t_matrix), axis=0)] = np.nan

    im = ax.imshow(
        sig_matrix,
        cmap="bwr",
        aspect="auto",
        vmin=-1,
        vmax=1,
        extent=[0, 360, 0, n_muscles],
        interpolation="nearest"
    )
    # ==============================
    # 4. Labels axes
    # ==============================

    ax.set_yticks(np.arange(n_muscles) + 0.5)
    ax.set_yticklabels(muscle_names_ord, fontsize=9)

    ax.set_xlabel("Pedal angle (°)", fontsize=12)
    ax.set_xticks(np.arange(0, 361, 60))

    ax.set_title("SPM significant clusters (ECC vs CON)", fontsize=14, weight="bold")

    ax_group.set_ylim(0, n_muscles)
    ax_group.set_xlim(0, 1)

    for start, end, group in group_info:
        y_center = (start + end) / 2

        ax_group.text(
            -0.2,
            y_center,
            format_group_label(group),
            rotation=0,
            va='center',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

        # séparation visuelle
        ax_group.hlines(end, 0, 1, colors='black', linewidth=1)
        ax.hlines(end, 0, 360, colors='black', linewidth=1)

    # enlever axes inutiles
    ax_group.axis('off')

    # ==============================
    # 5. Lignes groupes
    # ==============================

    for start, end, _ in group_info:
        ax.axhline(end, color="black", linewidth=1)


    # ==============================
    # 7. Légende simple
    # ==============================

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='red', label='ECC > CON'),
        Patch(facecolor='blue', label='CON > ECC'),
        Patch(facecolor='white', edgecolor='black', label='Non-significant')
    ]

    fig.legend(
        handles=legend_elements,
        loc='center right',
        bbox_to_anchor=(0.95, 0.95),
        frameon=False
    )

    # ==============================
    # 8. Style publication
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
            v_musc = np.load(f"{base_path}/{mode}_{power}W/vitesse_musculaire.npy")
            if mode=="eccentric":
                crank, q, act, frc, v_musc = ensure_forward_rotation(crank, q, act, frc, v_musc)
            cycles = build_all_muscle_cycles(act, crank, N_POINTS)
            cycles_v_musc = build_all_muscle_cycles(v_musc, crank, N_POINTS)
            data[key] = {"q": q, "act": act, "frc": frc, "crank": crank, "cycles": cycles, "cycles_v_musc": cycles_v_musc}
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

    # =============================
    # ANALYSE AVANCÉE
    # =============================

    pattern_diff = compute_pattern_difference(cycles_con, cycles_ecc)

    df_metrics["Pattern diff"] = pattern_diff

    print("\nTop muscles avec changement de pattern :")
    print(df_metrics.sort_values("Pattern diff", ascending=False).head(10))

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
        results_spm,
        muscle_names,
        MUSCLE_GROUPS,
        t_matrix,
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

    r_global, df_corr = compute_pattern_similarity_grouped(
        cycles_con,
        cycles_ecc,
        muscle_names,
        MUSCLE_GROUPS
    )

    rmse_global, df_rmse = compute_rmse_grouped(
        cycles_con,
        cycles_ecc,
        muscle_names,
        MUSCLE_GROUPS
    )

    #Corrélation et RMSE sur vitesse musculaires

    r_global_v, df_corr_v = compute_pattern_similarity_grouped(
        data[con_key]["cycles_v_musc"],
        data[ecc_key]["cycles_v_musc"],
        muscle_names,
        MUSCLE_GROUPS
    )

    rmse_global_v, df_rmse_v = compute_rmse_grouped(
        data[con_key]["cycles_v_musc"],
        data[ecc_key]["cycles_v_musc"],
        muscle_names,
        MUSCLE_GROUPS
    )
