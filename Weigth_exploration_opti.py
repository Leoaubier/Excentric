import numpy as np
import time
import casadi as ca
import biorbd
import biorbd_casadi as biorbdc
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings

MODE_PEDALAGE = "eccentric"
PUISSANCE = "40"

MODEL_PATH = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie_vtp.bioMod"

Q_PATH    = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/q_inverse_kinematic.npy"
QDOT_PATH = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/qdot_inverse_kinematic.npy"
TAU_PATH  = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/tau_inverse_dynamic.npy"
EMG_PATH  = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/emg_processed_resampled.npy"

FIRST, END = 3300, 3400
RES_BND_know = 2
RES_BND_unknow = 4
TAU_RES_BND = np.concatenate((
    RES_BND_unknow * np.ones(5),
    RES_BND_know   * np.ones(5)
))
EPS_ACT = 1e-6

W_TAU = 1e8
W_RES = 1e5
W_EMG = 1e9
W_ACT = 1e7

DELAY = 50 # en ms (EMG en avance sur activation) : facteur 10

active_dof = [6,7,8,9,10,11,12,13,14,15]


W_TAU = 2.4e10
W_RES = 1.4e7
W_EMG = 1.2e10
W_ACT = 1.9e8

DELAY = 50 # en ms (EMG en avance sur activation) : facteur 10

active_dof = [6,7,8,9,10,11,12,13,14,15]

emg_to_muscle = {
    0: "DeltoideusClavicle",
    1: "DeltoideusScapula_M",
    2: "DeltoideusScapula_P",
    3: "TrapeziusScapula_S",
    4: "TRI",
    5: "BIC",
    6: "TrapeziusScapula_M",
    7: "TrapeziusScapula_I",
    8: "LatissimusDorsi_I",
    9: "PectoralisMajorThorax_M",
    #10: "brachio" #pas dans le modèle
}


def transpose_if_needed(arr, target_rows):
    return arr if arr.shape[0] == target_rows else arr.T

def ms_to_frame(delay):
    n_frame = delay/10
    return n_frame

def zoom_around(bestW, n_per=8, span=1.0, seed=1):
    # span=1.0 => +/- 1 ordre de grandeur autour
    rng = np.random.default_rng(seed)
    out = []
    for (wt, wr, we, wa) in bestW:
        logs = np.log10([wt, wr, we, wa])
        for _ in range(n_per):
            jitter = rng.uniform(-span, span, size=4)
            ww = 10 ** (logs + jitter)
            out.append(tuple(ww))
    return out

def sample_log_uniform(n, low_exp, high_exp, rng=None):
    """
    Renvoie n valeurs tirées log-uniformément entre 10^low_exp et 10^high_exp.
    """
    rng = np.random.default_rng(rng)
    exps = rng.uniform(low_exp, high_exp, size=n)
    return 10 ** exps


def make_weight_sweep_random(n=50, seed=0,
                            tau_exp=(4, 12),
                            res_exp=(2, 10),
                            emg_exp=(4, 12),
                            act_exp=(2, 10)):
    rng = np.random.default_rng(seed)
    W_TAU = sample_log_uniform(n, *tau_exp, rng=rng)
    W_RES = sample_log_uniform(n, *res_exp, rng=rng)
    W_EMG = sample_log_uniform(n, *emg_exp, rng=rng)
    W_ACT = sample_log_uniform(n, *act_exp, rng=rng)
    return list(zip(W_TAU, W_RES, W_EMG, W_ACT))

def pareto_front(costs):
    """
    costs : (N, D) à minimiser (plus petit = meilleur)
    return: mask (N,) True si non-dominé
    """
    costs = np.asarray(costs, float)
    N = costs.shape[0]
    is_pareto = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_pareto[i]:
            continue
        # j domine i si costs[j] <= costs[i] partout et < sur au moins 1 dim
        dominates = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        dominates[i] = False
        if np.any(dominates):
            is_pareto[i] = False
    return is_pareto


def extract_cycles_generic(signal, peaks):
    out = []
    for i in range(len(peaks) - 1):
        seg = signal[peaks[i]:peaks[i + 1]]
        seg_norm = np.interp(
            np.linspace(0, 1, 200),
            np.linspace(0, 1, len(seg)),
            seg
        )
        out.append(seg_norm)
    return np.array(out)

def compute_metrics(tau, tau_musc, tau_res, mus_act, emg_full_series, is_emg_mask, TAU_RES_BND, track_idx):
    """
    tau, tau_musc, tau_res : (nbTau, n_frames)
    mus_act : (nbMus, n_frames)
    emg_full_series : (nbMus, n_frames) (déjà mis sur tous les muscles avec 0 ailleurs)
    is_emg_mask : (nbMus,) 1 si EMG dispo
    TAU_RES_BND : (nbTau,) bornes abs
    track_idx : indices muscles trackés (EMG)
    """
    eps = 1e-12
    tau_err = tau - (tau_musc + tau_res)  # (nbTau,nF)

    # ---- Couples ----
    tau_err_abs = np.abs(tau_err)
    tau_err_rms = float(np.sqrt(np.mean(tau_err**2)))
    tau_err_p95 = float(np.percentile(tau_err_abs, 95))

    tau_res_abs = np.abs(tau_res)
    tau_res_rms = float(np.sqrt(np.mean(tau_res**2)))
    # "proche borne" : proportion d'échantillons où |tau_res| > 90% borne
    near = (tau_res_abs > (0.9 * TAU_RES_BND.reshape(-1, 1))).mean()
    tau_res_nearBnd = float(near)

    # ---- Saturation activation ----
    sat_thr = 0.99
    sat_rate_global = float((mus_act > sat_thr).mean())
    sat_rate_by_mus = (mus_act > sat_thr).mean(axis=1)  # (nbMus,)

    # ---- Suivi EMG (sur muscles trackés seulement) ----
    # (emg_full_series contient les EMG aux bons muscles, 0 ailleurs)
    tracked_mask = np.zeros(mus_act.shape[0], dtype=bool)
    tracked_mask[track_idx] = True

    a_tr = mus_act[tracked_mask, :]
    e_tr = emg_full_series[tracked_mask, :]

    emg_rmse = float(np.sqrt(np.mean((a_tr - e_tr) ** 2)))

    # corr moyenne par muscle (robuste)
    cors = []
    for i in range(a_tr.shape[0]):
        a_i = a_tr[i]
        e_i = e_tr[i]
        if np.std(a_i) < 1e-6 or np.std(e_i) < 1e-6:
            continue
        c = np.corrcoef(a_i, e_i)[0, 1]
        if np.isfinite(c):
            cors.append(c)
    emg_corr = float(np.mean(cors)) if len(cors) else np.nan

    return {
        "tau_err_rms": tau_err_rms,
        "tau_err_p95": tau_err_p95,
        "tau_res_rms": tau_res_rms,
        "tau_res_nearBnd": tau_res_nearBnd,
        "sat_rate_global": sat_rate_global,
        "emg_rmse": emg_rmse,
        "emg_corr": emg_corr,
        # utile si tu veux classer ensuite
        "score_default": tau_err_rms + 0.5 * emg_rmse + 0.1 * tau_res_rms + 0.1 * sat_rate_global,
        "sat_rate_by_mus": sat_rate_by_mus,  # array
    }


def run_one_setting(
    solver, f_cost,
    model_np,
    q, qdot, tau, emg,
    track_idx, emg_idx, is_emg_mask,
    TAU_RES_BND,
    W_TAU, W_RES, W_EMG, W_ACT,
    active_dof,
    EPS_ACT=1e-6,
):
    """
    Retourne métriques + éventuellement séries (si tu veux les inspecter après).
    """
    nbMus = model_np.nbMuscles()
    nbTau = len(active_dof)
    n_frames = q.shape[1]

    mus_act  = np.zeros((nbMus, n_frames))
    tau_res  = np.zeros((nbTau, n_frames))
    tau_musc = np.zeros((nbTau, n_frames))

    # EMG remappé sur tous les muscles (pour métriques)
    emg_full_series = np.zeros((nbMus, n_frames))

    # bornes / init
    x0 = np.concatenate([0.1*np.ones(nbMus), np.zeros(nbTau)])
    lbx = np.concatenate([np.zeros(nbMus), -TAU_RES_BND])
    ubx = np.concatenate([np.ones(nbMus),  TAU_RES_BND])

    for k in range(n_frames):
        emg_full = np.zeros(nbMus)
        emg_full[track_idx] = emg[emg_idx, k]
        emg_full_series[:, k] = emg_full

        p = np.concatenate([
            q[:,k], qdot[:,k], tau[:,k],
            emg_full, is_emg_mask,
            [W_TAU, W_RES, W_EMG, W_ACT]
        ])

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, p=p)
        xopt = np.array(sol["x"]).squeeze()

        a_opt = np.clip(xopt[:nbMus], 0, 1-EPS_ACT)
        mus_act[:,k] = a_opt
        tau_res[:,k] = xopt[nbMus:]

        # tau_musc (pour tau_err)
        states = model_np.stateSet()
        for i in range(nbMus):
            states[i].setActivation(a_opt[i])

        tau_musc[:,k] = model_np.muscularJointTorque(
            states, q[:,k], qdot[:,k]
        ).to_array()[active_dof]

    metrics = compute_metrics(
        tau=tau, tau_musc=tau_musc, tau_res=tau_res,
        mus_act=mus_act, emg_full_series=emg_full_series,
        is_emg_mask=is_emg_mask,
        TAU_RES_BND=TAU_RES_BND,
        track_idx=track_idx
    )

    out = {
        "weights": (W_TAU, W_RES, W_EMG, W_ACT),
        "metrics": metrics,
        # garde si tu veux inspecter les meilleurs runs
        "series": {
            "mus_act": mus_act,
            "tau_res": tau_res,
            "tau_musc": tau_musc,
            "emg_full": emg_full_series,
        }
    }
    return out

def make_weight_grid():
    # à ajuster : 3 valeurs par poids -> 81 runs (peut être long)
    W_TAU_list = [1e6, 1e8, 1e10]
    W_RES_list = [1e3, 1e5, 1e7]
    W_EMG_list = [1e7, 1e9, 1e11]
    W_ACT_list = [1e5, 1e7, 1e9]

    grid = []
    for wt in W_TAU_list:
        for wr in W_RES_list:
            for we in W_EMG_list:
                for wa in W_ACT_list:
                    grid.append((wt, wr, we, wa))
    return grid

def summarize_and_plot(results, top_k=8):
    # Table arrays
    W = np.array([r["weights"] for r in results], dtype=float)
    tau_err = np.array([r["metrics"]["tau_err_rms"] for r in results])
    emg_rmse = np.array([r["metrics"]["emg_rmse"] for r in results])
    tau_res = np.array([r["metrics"]["tau_res_rms"] for r in results])
    sat = np.array([r["metrics"]["sat_rate_global"] for r in results])
    emg_corr = np.array([r["metrics"]["emg_corr"] for r in results])

    # --- Pour visualiser une "dimension poids" en couleur : ratio EMG/ACT ---
    color_val = np.log10(W[:, 2] / (W[:, 3] + 1e-30))  # log10(W_EMG/W_ACT)
    size_val = 30 + 200 * sat  # saturation -> taille

    # --- Pareto (sur 4 critères) ---
    costs = np.vstack([tau_err, emg_rmse, tau_res, sat]).T
    pareto = pareto_front(costs)

    # --- Score normalisé juste pour trier top_k (pas pour conclure) ---
    def z(x):
        x = np.asarray(x, float)
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

    score = z(tau_err) + z(emg_rmse) + 0.5 * z(tau_res) + 0.5 * z(sat)
    order = np.argsort(score)
    best = order[:top_k]

    # ============ Plot 1 : Trade-off principal ============
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(tau_err, emg_rmse, c=color_val, s=size_val, alpha=0.85)
    plt.xlabel("tau_err RMS (Nm)  ↓")
    plt.ylabel("EMG RMSE (a vs emg)  ↓")
    plt.title("Trade-off (couple vs EMG)\nCouleur=log10(W_EMG/W_ACT), Taille=Saturation")
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc)
    cbar.set_label("log10(W_EMG/W_ACT)")

    # Pareto en contour (points cerclés)
    plt.scatter(tau_err[pareto], emg_rmse[pareto], facecolors="none", edgecolors="k", s=120, linewidths=1.5,
                label="Pareto (non-dominé)")

    # Annotation top_k
    for rank, idx in enumerate(best, 1):
        plt.annotate(str(rank), (tau_err[idx], emg_rmse[idx]), textcoords="offset points", xytext=(6, 6))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # ============ Plot 2 : Résidu vs saturation ============
    plt.figure(figsize=(8, 6))
    sc2 = plt.scatter(tau_res, sat, c=color_val, s=60, alpha=0.85)
    plt.xlabel("tau_res RMS (Nm)  ↓")
    plt.ylabel("Saturation rate (a>0.99)  ↓")
    plt.title("Résidu vs saturation\nCouleur=log10(W_EMG/W_ACT)")
    plt.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(sc2)
    cbar2.set_label("log10(W_EMG/W_ACT)")
    plt.tight_layout()
    plt.show()

    # ============ Print clair Top K ============
    print("\nTOP candidats (labels = numéros sur le plot 1):")
    for rank, idx in enumerate(best, 1):
        wt, wr, we, wa = W[idx]
        print(
            f"[{rank}] W_TAU={wt:.1e}, W_RES={wr:.1e}, W_EMG={we:.1e}, W_ACT={wa:.1e} | "
            f"tau_err={tau_err[idx]:.3g} | emg_rmse={emg_rmse[idx]:.3g} | tau_res={tau_res[idx]:.3g} | "
            f"sat={sat[idx] * 100:.1f}% | emg_corr={emg_corr[idx]:.3g}"
        )

    # retourne indices utiles
    return {
        "best_idx": best,
        "pareto_idx": np.where(pareto)[0],
        "score": score
    }

def build_emg_to_muscle_mapping(model: biorbd.Model, emg_to_muscle_dict: dict, verbose=True):
    muscle_names = [model.muscle(i).name().to_string() for i in range(model.nbMuscles())]
    sorted_items = sorted(emg_to_muscle_dict.items(), key=lambda x: x[0])

    track_idx = []
    emg_src_idx = []
    tracked_muscle_names = []

    if verbose:
        print("\n[EMG → Muscle mapping]")
        print("-" * 60)

    for emg_ch, key in sorted_items:
        matches = [i for i, mname in enumerate(muscle_names) if key in mname]
        if len(matches) == 0:
            raise ValueError(f"No muscle matched for EMG key '{key}' (ch {emg_ch})")

        if verbose:
            print(f"EMG {emg_ch:>2} ({key:25s}) → {[muscle_names[i] for i in matches]}")

        for mi in matches:
            track_idx.append(mi)
            emg_src_idx.append(emg_ch)
            tracked_muscle_names.append(muscle_names[mi])

    return np.array(track_idx, dtype=int), np.array(emg_src_idx, dtype=int), tracked_muscle_names

def build_nlp_solver(model_path, nb_mus, nb_tau):
    model = biorbdc.Model(model_path)

    a = ca.MX.sym("a", nb_mus)
    tau_res = ca.MX.sym("tau_res", nb_tau)
    x = ca.vertcat(a, tau_res)

    q     = ca.MX.sym("q", model.nbQ())
    qdot  = ca.MX.sym("qdot", model.nbQ())
    tauID = ca.MX.sym("tauID", nb_tau)
    emg   = ca.MX.sym("emg", nb_mus)
    mask  = ca.MX.sym("mask", nb_mus)

    w_tau, w_res, w_emg, w_act = ca.MX.sym("w_tau"), ca.MX.sym("w_res"), ca.MX.sym("w_emg"), ca.MX.sym("w_act")

    p = ca.vertcat(q, qdot, tauID, emg, mask, w_tau, w_res, w_emg, w_act)

    states = model.stateSet()
    for i in range(model.nbMuscles()):
        states[i].setActivation(a[i])

    tau_m_full = model.muscularJointTorque(states, q, qdot).to_mx()
    tau_m = tau_m_full[active_dof]

    tau_err = tauID - (tau_m + tau_res)
    emg_err = (emg - a) * mask
    a_free  = a * (1 - mask)

    cost = (
        w_tau * ca.sumsqr(tau_err) +
        w_res * ca.sumsqr(tau_res) +
        w_emg * ca.sumsqr(emg_err) +
        w_act * ca.sumsqr(a_free) +
        ca.sum(a**9)
    )

    solver = ca.nlpsol(
        "solver", "ipopt",
        {"x": x, "f": cost, "p": p},
        {"ipopt.print_level": 5, "print_time": True}
    )

    f_cost = ca.Function("f_cost", [x, p], [cost])

    return solver, f_cost

def main():

    # ==========================================================
    # 1) LOAD DATA & PREP (UNE SEULE FOIS)
    # ==========================================================
    model_np = biorbd.Model(MODEL_PATH)
    nbQ   = model_np.nbQ()
    nbMus = model_np.nbMuscles()
    nbTau = len(active_dof)

    q    = np.load(Q_PATH)
    qdot = np.load(QDOT_PATH)
    tau  = np.load(TAU_PATH)
    emg  = np.load(EMG_PATH)

    q    = q[:, FIRST:END]
    qdot = qdot[:, FIRST:END]
    tau  = tau[active_dof, FIRST:END]
    emg  = emg[:, int(FIRST+ms_to_frame(DELAY)):int(END+ms_to_frame(DELAY))]

    print("q shape   :", q.shape)
    print("emg shape :", emg.shape)

    track_idx, emg_idx, tracked_names = build_emg_to_muscle_mapping(model_np, emg_to_muscle)

    is_emg_mask = np.zeros(nbMus)
    is_emg_mask[track_idx] = 1.0

    # IMPORTANT : solver construit UNE SEULE FOIS
    solver, f_cost = build_nlp_solver(MODEL_PATH, nbMus, nbTau)

    # ==========================================================
    # 2) PHASE 1 — LARGE RANDOM SWEEP (log-uniform)
    # ==========================================================
    print("\n================ PHASE 1 : LARGE SWEEP ================\n")
    grid1 = make_weight_sweep_random(
        n=60, seed=0,
        tau_exp=(4, 12),
        res_exp=(2, 10),
        emg_exp=(4, 12),
        act_exp=(2, 10),
    )

    results1 = []
    t0 = time.time()

    for it, (wt, wr, we, wa) in enumerate(grid1, 1):
        print(f"[P1] Run {it}/{len(grid1)} | {wt:.1e} {wr:.1e} {we:.1e} {wa:.1e}")

        out = run_one_setting(
            solver, f_cost,
            model_np,
            q, qdot, tau, emg,
            track_idx, emg_idx, is_emg_mask,
            TAU_RES_BND,
            W_TAU=wt, W_RES=wr, W_EMG=we, W_ACT=wa,
            active_dof=active_dof,
            EPS_ACT=EPS_ACT,
        )
        results1.append(out)

    print("\nPhase 1 done in", time.time() - t0, "s")

    info1 = summarize_and_plot(results1, top_k=10)

    # ==========================================================
    # 3) PHASE 2 — ZOOM AUTOUR DES MEILLEURS
    # ==========================================================
    print("\n================ PHASE 2 : ZOOM ================\n")

    bestW = [results1[i]["weights"] for i in info1["best_idx"][:5]]

    grid2 = zoom_around(bestW, n_per=10, span=1.0, seed=1)

    results2 = []
    t0 = time.time()

    for it, (wt, wr, we, wa) in enumerate(grid2, 1):
        print(f"[P2] Run {it}/{len(grid2)} | {wt:.1e} {wr:.1e} {we:.1e} {wa:.1e}")

        out = run_one_setting(
            solver, f_cost,
            model_np,
            q, qdot, tau, emg,
            track_idx, emg_idx, is_emg_mask,
            TAU_RES_BND,
            W_TAU=wt, W_RES=wr, W_EMG=we, W_ACT=wa,
            active_dof=active_dof,
            EPS_ACT=EPS_ACT,
        )
        results2.append(out)

    print("\nPhase 2 done in", time.time() - t0, "s")

    # ==========================================================
    # 4) ANALYSE FINALE (phase1 + phase2 ensemble)
    # ==========================================================
    all_results = results1 + results2
    summarize_and_plot(all_results, top_k=12)


if __name__ == "__main__":
    main()
