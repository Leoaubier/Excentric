import numpy as np
import time
import biorbd
import casadi as ca
import matplotlib.pyplot as plt

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod"

Q_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy"
QDOT_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/qdot_inverse_kinematic_sidonie_40W.npy"
TAU_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/ID/tau_inverse_dynamic_Sidonie_40w.npy"
EMG_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/EMG/emg_processed_resampled.npy"

# ----------------------------
# Config
# ----------------------------
FIRST, END = 2000, 3000
EPS_ACT = 1e-6

TAU_RES_BND = 5.0  # ±5 Nm as in Ceglia et al.

# Suggested starting weights (tune)
W_TAU = 5e3        # torque tracking
W_RES = 1e2        # residual torque penalty
W_EMG = 20.0       # EMG tracking
W_ACT = 1e-3       # activation penalty for non-EMG muscles

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
# EMG mapping (substring matching)
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
def get_moment_arms_and_fmax(model, q):
    q = np.asarray(q).reshape(-1)
    nb_q = model.nbQ()
    nb_mus = model.nbMuscles()

    J = model.musclesLengthJacobian(q).to_array()  # (nbMus, nbQ)
    if J.shape != (nb_mus, nb_q):
        raise RuntimeError(f"Jacobian shape {J.shape}, expected {(nb_mus, nb_q)}")

    R = -J.T  # (nbQ, nbMus)

    Fmax = np.array([model.muscle(i).characteristics().forceIsoMax() for i in range(nb_mus)], dtype=float)
    return R, Fmax

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
    emg_err = (a - emg) * is_emg

    # activation regularization only for muscles WITHOUT EMG
    a_ninf = a * (1 - is_emg)

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
    nbTau = model.nbGeneralizedTorque()
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
    emg_env = np.clip(emg_env / 100.0, 0.0, 1.0)

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
    tau = tau[:, FIRST:END]
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
    is_emg_mask = np.ones((nbTrackedMus,), dtype=float)

    # Solver
    solver = build_ceglia_solver_with_p(nbTrackedMus, nbTau, qp_solver_name=QP_SOLVER)

    # Outputs
    mus_act = np.zeros((nbTrackedMus, n_frames))
    tau_res = np.zeros((nbTau, n_frames))
    tau_err = np.zeros((nbTau, n_frames))
    mus_force = np.zeros((nbMus, n_frames))
    emg_used = np.zeros((nbTrackedMus, n_frames))

    # Floating base: zero first 6 dof if needed
    if nbQ == 18:
        q[:6, :] = 0.0
        qdot[:6, :] = 0.0
        tau[:6, :] = 0.0

    t0 = time.time()
    for k in range(n_frames):
        qk = q[:, k].reshape(-1)
        qdotk = qdot[:, k].reshape(-1)
        tauk = tau[:, k].reshape(-1)

        # Build A
        R, Fmax = get_moment_arms_and_fmax(model, qk)
        R_tr = R[:, track_idx]                         # (nbTau, nbTrackedMus)
        F_tr = Fmax[track_idx]                         # (nbTrackedMus,)
        A = R_tr * F_tr.reshape(1, -1)                 # (nbTau, nbTrackedMus)

        # EMG duplicated per tracked muscle
        emgk = emg_env[:, k].reshape(-1)
        emg_tr = emgk[emg_src_idx]

        # Solve Ceglia QP
        a_tr, tau_res_tr = run_ceglia_frame(
            solver=solver,
            A_np=A,
            tau_np=tauk,
            emg_np=emg_tr,
            is_emg_mask_np=is_emg_mask,
            w_tau_val=W_TAU,
            w_res_val=W_RES,
            w_emg_val=W_EMG,
            w_act_val=W_ACT,
        )

        # Compute errors
        tau_m = A @ a_tr
        tau_err_k = tauk - (tau_m + tau_res_tr)  # should be small when W_TAU high

        # Full activations for force computation
        a_full = np.zeros((nbMus,), dtype=float)
        a_full[track_idx] = a_tr
        f_full = compute_muscle_forces_from_activation(model, qk, qdotk, a_full)

        # Save
        mus_act[:, k] = a_tr
        tau_res[:, k] = tau_res_tr
        tau_err[:, k] = tau_err_k
        mus_force[:, k] = f_full
        emg_used[:, k] = emg_tr

        if (k + 1) % 200 == 0:
            print(f"Frame {k+1}/{n_frames} done.")

    total = time.time() - t0
    print("\nDone.")
    print(f"Total time: {total:.3f} s")

    # ----------------------------
    # Diagnostics plots
    # ----------------------------
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

    plt.figure(figsize=(14, 6))
    for i in range(nbTrackedMus):
        plt.plot(mus_act[i, :], label=f"{i}: {tracked_names[i]}")
    plt.title("Activations musculaires (Ceglia EMG-informed SO)")
    plt.xlabel("Frame")
    plt.ylabel("Activation")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

    # Return results if you want to save
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
