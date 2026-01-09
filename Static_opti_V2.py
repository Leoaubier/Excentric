import numpy as np
import time
import biorbd
import casadi as ca
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.BioMod"

Q_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy"
QDOT_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/qdot_inverse_kinematic_sidonie_40W.npy"
TAU_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/ID/tau_inverse_dynamic_Sidonie_40w.npy"
EMG_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/EMG/emg_processed_resampled.npy"

# ----------------------------
# Config
# ----------------------------
FIRST, END = 3000, 6000
EPS_ACT = 1e-4 # éviter une activation à 0 ou 1

TAU_RES_BND = 0.1  # ±5 Nm as in Ceglia et al.


W_EMG = 0    # EMG tracking
W_ACT = 10000000    # activation penalty for non-EMG muscles
W_TAU = 10000000  # torque tracking
W_RES = 1       # residual torque penalty

#active_dof = [6,7,8,9,10,11,12,13,14,15]
active_dof = [8,9,10,11,12,13,14]



QP_SOLVER = "qpoases"  # fallback to osqp below


# ----------------------------
# EMG -> muscle mapping
# ----------------------------
emg_to_muscle = {
    0: "DeltoideusClavicle",
    1: "DeltoideusScapula_M",
    2: "DeltoideusScapula_P",
    3: "TRI_",
    4: "BIC_",
    5: "TrapeziusScapula_M",
    6: "TrapeziusScapula_S",
    7: "TrapeziusScapula_I",
    8: "LatissimusDorsi",
    9: "PectoralisMajor",
}

# ----------------------------
# Helpers: shape / IO
# ----------------------------
def _assert_2d(name: str, arr: np.ndarray):
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(arr)}")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got shape={arr.shape}")

def _maybe_transpose_to_dof_by_frames(arr: np.ndarray, expected_dof: int, name: str) -> np.ndarray:
    _assert_2d(name, arr)
    if arr.shape[0] == expected_dof:
        return arr
    if arr.shape[1] == expected_dof:
        return arr.T
    raise ValueError(f"{name} incompatible shape={arr.shape}, expected dof={expected_dof}")

def _maybe_transpose_frames_match(arr: np.ndarray, n_frames: int, name: str) -> np.ndarray:
    _assert_2d(name, arr)
    if arr.shape[1] == n_frames:
        return arr
    if arr.shape[0] == n_frames:
        return arr.T
    raise ValueError(f"{name} frames mismatch: {arr.shape} vs n_frames={n_frames}")

# ----------------------------
# EMG mapping
# ----------------------------

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

# ----------------------------
# Biorbd quantities
# ----------------------------
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

def get_R_and_Fiso(model, q):
    nb_mus = model.nbMuscles()


    # Jacobien des longueurs musculaires
    J = model.musclesLengthJacobian(q).to_array()   # (nbMus, nbQ)
    R = -J.T                                        # (nbQ, nbMus) --> -J normalement

    # Force isométrique maximale (bornée, physiologique)
    Fiso = np.array(
        [model.muscle(i).characteristics().forceIsoMax() for i in range(nb_mus)],
        dtype=float
    )

    return R, Fiso


def compute_muscle_forces_from_activation(model: biorbd.Model, q: np.ndarray, qdot: np.ndarray, a: np.ndarray):
    a = np.asarray(a, dtype=float).reshape(-1)
    nb_mus = model.nbMuscles()
    if a.shape[0] != nb_mus:
        raise ValueError(f"Activation size {a.shape[0]} != nbMuscles {nb_mus}")

    states = model.stateSet()
    for i in range(nb_mus):
        states[i].setActivation(float(a[i]))

    forces = model.muscleForces(states, q, qdot).to_array()
    return np.asarray(forces).reshape(-1)

# ----------------------------
# Ceglia QP: variables = [a ; tau_res]
# ----------------------------
def build_ceglia_solver_with_p(nb_mus: int, nb_tau: int, qp_solver_name=QP_SOLVER):
    """
    Decision variables:
      x = [a (nb_mus) ; tau_res (nb_tau)]
    Parameters:
      p = [vec(A) ; tau ; emg ; is_emg_mask ; w_tau ; w_res ; w_emg ; w_act]
    where:
      A: (nb_tau, nb_mus)
      tau: (nb_tau,)
      emg: (nb_mus,)   (already duplicated to match tracked muscles)
      is_emg_mask: (nb_mus,) 1 if muscle has EMG, else 0
    """
    x = ca.MX.sym("x", nb_mus + nb_tau)
    a = x[:nb_mus]
    tau_res = x[nb_mus:]

    p = ca.MX.sym("p", nb_tau * nb_mus + nb_tau + nb_mus + nb_mus + 4)

    off = 0
    A_vec = p[off: off + nb_tau * nb_mus]; off += nb_tau * nb_mus
    tau = p[off: off + nb_tau]; off += nb_tau
    emg = p[off: off + nb_mus]; off += nb_mus
    is_emg = p[off: off + nb_mus]; off += nb_mus
    w_tau = p[off]; w_res = p[off + 1]; w_emg = p[off + 2]; w_act = p[off + 3]

    A = ca.reshape(A_vec, nb_tau, nb_mus)
    tau_m = A @ a

    # torque tracking: tau - (tau_m + tau_res)
    tau_err = tau - (tau_m + tau_res)

    # EMG term only for muscles with EMG
    emg_err = (emg - a) * is_emg

    # activation regularization only for muscles WITHOUT EMG
    #a_ninf = a * (1 - is_emg)
    a_ninf = a #car wemg est à 0
    cost = (
        w_tau * ca.sumsqr(tau_err)
        + w_res * ca.sumsqr(tau_res)
        + w_emg * ca.sumsqr(emg_err)
        + w_act * ca.sumsqr(a_ninf)
    )

    qp = {"x": x, "f": cost, "g": ca.MX(), "p": p}

    try:
        solver = ca.qpsol("ceglia_qp", qp_solver_name, qp)
    except Exception:
        solver = ca.qpsol("ceglia_qp", "osqp", qp)

    return solver

def run_ceglia_frame(
    solver,
    A_np: np.ndarray,
    tau_np: np.ndarray,
    emg_np: np.ndarray,
    is_emg_mask_np: np.ndarray,
    w_tau_val: float,
    w_res_val: float,
    w_emg_val: float,
    w_act_val: float,
):
    nb_tau, nb_mus = A_np.shape

    p = np.concatenate([
        A_np.reshape(-1),
        tau_np.reshape(-1),
        emg_np.reshape(-1),
        is_emg_mask_np.reshape(-1),
        np.array([w_tau_val, w_res_val, w_emg_val, w_act_val], dtype=float),
    ])

    # Bounds: 0<=a<=1 ; -5<=tau_res<=5
    lbx = np.concatenate([np.zeros(nb_mus), -TAU_RES_BND * np.ones(nb_tau)])
    ubx = np.concatenate([np.ones(nb_mus),  TAU_RES_BND * np.ones(nb_tau)])

    sol = solver(lbx=lbx, ubx=ubx, p=p)
    x_opt = np.array(sol["x"]).reshape(-1)

    a_opt = x_opt[:nb_mus]
    tau_res_opt = x_opt[nb_mus:]

    a_opt = np.clip(a_opt, EPS_ACT, 1.0 - EPS_ACT)
    return a_opt, tau_res_opt

# ----------------------------
# Main
# ----------------------------
def main():
    model = biorbd.Model(MODEL_PATH)

    nbQ = model.nbQ()
    #nbTau = model.nbGeneralizedTorque()
    nbTau = len(active_dof)
    nbMus = model.nbMuscles()


    print("Model loaded.")
    print(f"  nbQ   = {nbQ}")
    print(f"  nbTau = {nbTau}")
    print(f"  nbMus = {nbMus}")

    # Load
    q = np.load(Q_PATH)
    qdot = np.load(QDOT_PATH)
    tau = np.load(TAU_PATH)
    emg_env = np.load(EMG_PATH)

    # EMG 0..100 -> 0..1
    emg_env = np.clip(emg_env, 0.0, 1.0)

    # Shapes
    q = _maybe_transpose_to_dof_by_frames(q, nbQ, "q")
    qdot = _maybe_transpose_to_dof_by_frames(qdot, nbQ, "qdot")
    tau = _maybe_transpose_frames_match(tau, q.shape[1], "tau")
    emg_env = _maybe_transpose_frames_match(emg_env, q.shape[1], "emg_env")

    # Window
    if not (0 <= FIRST < END <= q.shape[1]):
        raise ValueError(f"Invalid window FIRST={FIRST}, END={END}, nFrames={q.shape[1]}")

    q = q[:, FIRST:END]
    qdot = qdot[:, FIRST:END]
    tau = tau[active_dof, FIRST:END]
    emg_env = emg_env[:, FIRST:END]
    n_frames = q.shape[1]
    print(f"Data windowed: nFrames={n_frames}")

    # Mapping
    track_idx, emg_src_idx, tracked_names = build_emg_to_muscle_mapping(model, emg_to_muscle, verbose=True)
    if np.max(emg_src_idx) >= emg_env.shape[0]:
        raise ValueError(f"emg_env has {emg_env.shape[0]} channels but mapping requests {np.max(emg_src_idx)}")

    nbTrackedMus = track_idx.shape[0]
    print(f"\nTracked muscles (expanded): {nbTrackedMus}")

    # All tracked muscles have EMG (because they come from mapping) -> mask = 1
    # If you later decide to include more muscles without EMG, set those entries to 0
    # is_emg_mask = 1 uniquement pour les muscles réellement associés à un EMG
    is_emg_mask_full = np.zeros(nbMus, dtype=float)
    is_emg_mask_full[track_idx] = 1.0  # seulement les muscles avec EMG

    # Solver
    solver = build_ceglia_solver_with_p(nbMus, nbTau, qp_solver_name=QP_SOLVER)

    # Outputs
    mus_act = np.zeros((nbMus, n_frames))
    tau_res = np.zeros((nbTau, n_frames))
    tau_err = np.zeros((nbTau, n_frames))
    mus_force = np.zeros((nbMus, n_frames))
    emg_used = np.zeros((nbMus, n_frames))
    Fiso_musc = np.zeros((nbMus, n_frames))
    R_musc = np.zeros((nbTau,nbMus, n_frames))
    tau_act = np.zeros((nbTau, n_frames))

    scaling_factor_tau = 1.0  # normalisation globale du tau
    scaling_factor_act = 10.0  # pour forcer activations réalistes
    scaling_factor_emg = 100.0  # pour suivre EMG si W_EMG > 0

    # Pondération par DoF pour équilibrer les contributions
    tau_weight_per_dof = np.maximum(np.abs(tau[:, :n_frames].mean(axis=1)), 1.0)  # éviter 0
    tau_weight = 1.0 / tau_weight_per_dof  # plus le DoF est grand, moins il pèse

    t0 = time.time()
    for k in range(n_frames):
        qk = q[:, k].reshape(-1)
        qdotk = qdot[:, k].reshape(-1)
        tauk = tau[:, k].reshape(-1)

        # Build A
        R, Fiso = get_R_and_Fiso(model, qk)

        A_all = R * Fiso.reshape(1, -1)
        A = A_all[active_dof, :]

        # Scaling par DoF pour éviter qu'un DoF domine
        tauk_scaled = tauk * tau_weight
        A_scaled = A * tau_weight[:, None]

        # EMG seulement pour les muscles mappés
        emg_full = np.zeros(nbMus)
        emgk = emg_env[:, k].reshape(-1)
        emg_full[track_idx] = emgk[emg_src_idx]

        # Solve Ceglia QP avec pondérations et bornes physiologiques
        a_tr, tau_res_tr = run_ceglia_frame(
            solver=solver,
            A_np=A_scaled,
            tau_np=tauk_scaled,
            emg_np=emg_full * scaling_factor_emg,
            is_emg_mask_np=is_emg_mask_full,
            w_tau_val=W_TAU * scaling_factor_tau,
            w_res_val=W_RES,
            w_emg_val=W_EMG,
            w_act_val=W_ACT * scaling_factor_act,
        )

        # Compute errors
        tau_m = (A @ a_tr)

        print("Max |R|      :", np.max(np.abs(R)))
        print("Max f_tr    :", np.max(np.abs(Fiso)))
        print("Max |A|     :", np.max(np.abs(A)))
        print("Max |tau_m| :", np.max(np.abs(A @ a_tr)))
        print("Max |tau|   :", np.max(np.abs(tauk)))

        tau_err_k = tauk - (tau_m + tau_res_tr)  # should be small when W_TAU high

        f_full = compute_muscle_forces_from_activation(model, qk, qdotk, a_tr)
        #f_full = np.zeros(nbMus)
        # Save
        mus_act[:, k] = a_tr
        tau_act[:, k] = tau_m
        tau_res[:, k] = tau_res_tr
        tau_err[:, k] = tau_err_k
        mus_force[:, k] = f_full
        emg_used[:, k] = emg_full
        Fiso_musc[:, k] = Fiso
        R_musc[:, :, k] = R[active_dof]

        if (k + 1) % 200 == 0:
            print(f"Frame {k+1}/{n_frames} done.")

    total = time.time() - t0
    print("\nDone.")
    print(f"Total time: {total:.3f} s")

    np.save("/Users/leo/Desktop/Projet/Collecte_25_11/statique/muscle_activations.npy", mus_act)
    print("activations save")
    # ----------------------------
    # Plots
    # ----------------------------
    #for dof_idx in range(len(active_dof)):
    #    plt.figure(figsize=(10, 5))

    #    for m in range(nbMus):
    #        plt.plot(R_musc[dof_idx, m, :], alpha=0.4)
    #
    #    plt.axhline(0, color='k', linewidth=0.5)
    #    plt.title(f"Moment arms R(t) – DoF {active_dof[dof_idx]}")
    #    plt.xlabel("Frame")
    #    plt.ylabel("R (m)")
    #    plt.show()


    err_mean = np.mean(np.abs(tau_err), axis=1)
    res_mean = np.mean(np.abs(tau_res), axis=1)

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(nbTau), err_mean)
    plt.title("Moyenne de |tau_err| par DoF  (τ - (A a + τ_res))")
    plt.xlabel("DoF index")
    plt.ylabel("mean |tau_err|")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(nbTau), res_mean)
    plt.title("Moyenne de |τ_res| par DoF  (borne ±5 Nm)")
    plt.xlabel("DoF index")
    plt.ylabel("mean |τ_res|")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

    mus_act_emg = np.zeros((nbTrackedMus,n_frames))
    for i, idx in enumerate(track_idx):
        mus_act_emg[i, :] = mus_act[idx, :]
    plt.figure(figsize=(14, 6))
    for i in range(nbTrackedMus):
        plt.plot(mus_act_emg[i, :], label=f"{i}: {tracked_names[i]}")
    plt.title("Activations musculaires")
    plt.xlabel("Frame")
    plt.ylabel("Activation")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


    all_muscle_names = [model.muscle(i).name().to_string() for i in range(model.nbMuscles())]

    print("Activations par muscle (min / max) :")
    for i in range(mus_act.shape[0]):
        min_a = np.min(mus_act[i, :])
        max_a = np.max(mus_act[i, :])
        print(f"{i:2d} : {all_muscle_names[i]:30s} -> min={min_a:.4f}, max={max_a:.4f}")

    print("Fiso (min / max) :")
    for i in range(mus_act.shape[0]):
        min_Fiso = np.min(Fiso_musc[i, :])
        max_Fiso = np.max(Fiso_musc[i, :])
        print(f"{i:2d} : {all_muscle_names[i]:30s} -> min={min_Fiso:.4f}, max={max_Fiso:.4f}")

    print("R (min / max) :")
    for i in range(mus_act.shape[0]):
        min_R = np.min(R_musc[:, i, :])
        max_R = np.max(R_musc[:, i, :])
        print(f"{i:2d} : {all_muscle_names[i]:30s} -> min={min_R:.4f}, max={max_R:.4f}")

    plt.figure(figsize=(14, 6))
    for i in range(nbMus):
        plt.plot(mus_act[i, :], label=f"{i}: {all_muscle_names[i]}")
    plt.title("Activations musculaires")
    plt.xlabel("Frame")
    plt.ylabel("Activation")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    for i in range(nbMus):
        plt.plot(mus_force[i, :], label=f"{i}: {all_muscle_names[i]}")
    plt.title("Forces musculaires")
    plt.xlabel("Frame")
    plt.ylabel("Force en N")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))

    for i, name in enumerate(tracked_names):
        emg_ch = emg_src_idx[i]  # ✅ canal EMG correspondant à ce muscle

        plt.plot(
            mus_act_emg[i, :],
            label=f"Activation {name}"
        )

        plt.plot(
            emg_env[emg_ch,:],
            '--',
            label=f"EMG ch{emg_ch}"
        )

    plt.xlabel("Frame")
    plt.ylabel("Activation / EMG")
    plt.title("Activations musculaires vs EMG")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

    import plotly.graph_objects as go

    fig = go.Figure()

    for i in range(nbTau):
        # tau
        fig.add_trace(
            go.Scatter(
                y=tau[i, :],
                mode="lines",
                name=f"{i}: (tau)",
                legendgroup=f"group_{i}"
            )
        )

        # tau_res
        fig.add_trace(
            go.Scatter(
                y=tau_res[i, :],
                mode="lines",
                name=f"{i}: tau_res",
                legendgroup=f"group_{i}",
                line=dict(dash="dash")
            )
        )

        # tau_err
        fig.add_trace(
            go.Scatter(
                y=tau_act[i, :],
                mode="lines",
                name=f"{i}: tau_act",
                legendgroup=f"group_{i}",
                line=dict(dash="dot")
            )
        )

    fig.update_layout(
        title="Couples articulaires : tau, tau_res, tau_err",
        xaxis_title="Frame",
        yaxis_title="Couple (N·m)",
        hovermode="x unified",
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        template="plotly_white",
        height=500
    )

    fig.show()
    # Return results

    #---------- Plot final -----------
    # ==========================================================
    # DÉTECTION DES CYCLES À PARTIR D’UN DOF DE RÉFÉRENCE
    # ==========================================================

    # Choix automatique d’un DOF de référence pour détecter les cycles
    ref_idx = 0 #8 si les 10 DoF

    print(f"DoF utilisé comme référence du cycle : coude flexion")

    # Signal de référence
    ref_signal = tau[ref_idx, :]

    # Détection des peaks
    peaks_sel, _ = find_peaks(ref_signal, distance=100)

    print("Nombre de cycles détectés :", len(peaks_sel) - 1)
    # ==========================================================
    # === SUBPLOT : COUPLES / FORCES PAR DOF SUR LE CYCLE ===
    # =========================================================
    MUSC_TO_PLOT = "ALL"
    # ----------- Sélection DoF à tracer ------------
    if MUSC_TO_PLOT == "ALL":
        selected_musc = all_muscle_names
    else:
        selected_musc = MUSC_TO_PLOT

    # ----------- Construction cycles τ par DoF --------
    cycles_tau = {}
    mean_tau = {}
    std_tau = {}

    for dof in selected_musc:
        idx = all_muscle_names.index(dof)
        cyc = extract_cycles_generic(mus_act[idx, :], peaks_sel)
        cycles_tau[dof] = cyc
        mean_tau[dof] = np.mean(cyc, axis=0)
        std_tau[dof] = np.std(cyc, axis=0)
    # ----------- Plot final : GRILLE DE SUBPLOTS -------------------------

    import math

    x = np.linspace(0, 100, 200)

    # Définition automatique d'une grille
    n_cols = math.ceil(math.sqrt(nbMus))
    n_rows = math.ceil(nbMus / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axes = axes.flatten()  # pour parcourir facilement

    for ax, dof in zip(axes, all_muscle_names):

        # cycles individuels
        for c in cycles_tau[dof]:
            ax.plot(x, c, color="gray", alpha=0.25)

        # moyenne
        ax.plot(x, mean_tau[dof], linewidth=2, color="blue")

        # écart-type
        ax.fill_between(
            x,
            mean_tau[dof] - std_tau[dof],
            mean_tau[dof] + std_tau[dof],
            color="blue",
            alpha=0.15
        )

        ax.set_title(dof, fontsize=8)
        ax.set_ylabel("Activation")
        ax.grid(True)

    # Supprimer les axes inutilisés si la grille est trop grande
    for i in range(len(selected_musc), len(axes)):
        fig.delaxes(axes[i])

    # Label global
    plt.xlabel("Cycle (%)")
    plt.tight_layout()
    plt.show()

    return {
        "mus_act": mus_act,
        "tau_res": tau_res,
        "tau_err": tau_err,
        "mus_force": mus_force,
        "emg_proc": emg_used,
        "track_idx": track_idx,
        "tracked_names": tracked_names,
    }


if __name__ == "__main__":
    main()
