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
FIRST, END = 2000, 3000  # window frames
EPS_ACT = 1e-6

# Weights (simple, tune if needed)
W_TAU = 1.0       # torque tracking
W_ACT = 1e-3      # activation regularization
W_EMG = 50.0      # emg tracking (set 0.0 to disable)

# QP solver (qpoases is usually available via casadi; fallback is osqp)
QP_SOLVER = "qpoases"  # try: "qpoases" then fallback to "osqp"


# ----------------------------
# EMG -> muscle mapping
# (keys = EMG channel indices in your emg_env,
#  values = substring to match in model muscle names)
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
        raise ValueError(f"{name} must be 2D (nDoF x nFrames). Got shape={arr.shape}")


def _maybe_transpose_to_dof_by_frames(arr: np.ndarray, expected_dof: int, name: str) -> np.ndarray:
    """
    Ensures array is shaped (expected_dof, nFrames).
    If it looks like (nFrames, expected_dof), transpose it.
    """
    _assert_2d(name, arr)

    if arr.shape[0] == expected_dof:
        return arr
    if arr.shape[1] == expected_dof:
        return arr.T

    raise ValueError(
        f"{name} has incompatible shape {arr.shape}. "
        f"Expected ({expected_dof}, nFrames) or (nFrames, {expected_dof})."
    )


def _maybe_transpose_frames_match(arr: np.ndarray, n_frames: int, name: str) -> np.ndarray:
    """
    Ensures array second dimension is n_frames.
    If arr first dimension is n_frames, transpose.
    """
    _assert_2d(name, arr)
    if arr.shape[1] == n_frames:
        return arr
    if arr.shape[0] == n_frames:
        return arr.T
    raise ValueError(f"{name} frames mismatch: {arr.shape} vs expected n_frames={n_frames}")


# ----------------------------
# EMG mapping like the old code (substring matching)
# ----------------------------
def build_emg_to_muscle_mapping(model: biorbd.Model, emg_to_muscle_dict: dict, verbose=True):
    """
    Returns:
      - track_idx: indices of model muscles to track/optimize (length = nbTrackedMuscles)
      - emg_src_idx: for each tracked muscle, which EMG channel index to use (same length)
      - muscle_names_tracked: tracked muscle names in model order (same length)
    Notes:
      If one EMG key matches multiple muscles (e.g., "TRI_"), it duplicates the EMG channel
      for each matched muscle (same as your old approach).
    """
    muscle_names = [model.muscle(i).name().to_string() for i in range(model.nbMuscles())]

    # stable order by EMG channel index
    sorted_items = sorted(emg_to_muscle_dict.items(), key=lambda x: x[0])

    track_idx = []
    emg_src_idx = []
    muscle_names_tracked = []

    if verbose:
        print("\n[EMG → Muscle mapping]")
        print("-" * 60)

    for emg_ch, key in sorted_items:
        matches = [i for i, mname in enumerate(muscle_names) if key in mname]

        if len(matches) == 0:
            raise ValueError(
                f"No muscle matched for EMG key '{key}' (channel {emg_ch}).\n"
                f"Check naming vs model muscles."
            )

        if verbose:
            print(f"EMG {emg_ch:>2} ({key:25s}) → {[muscle_names[i] for i in matches]}")

        for mi in matches:
            track_idx.append(mi)
            emg_src_idx.append(emg_ch)
            muscle_names_tracked.append(muscle_names[mi])

    return (
        np.array(track_idx, dtype=int),
        np.array(emg_src_idx, dtype=int),
        muscle_names_tracked,
    )


# ----------------------------
# Biorbd quantities for SO
# ----------------------------
def get_moment_arms_and_fmax(model, q):
    """
    Parameters
    ----------
    model : biorbd.Model
    q     : (nbQ,) or (nbQ,1)

    Returns
    -------
    R    : (nbQ, nbMuscles) moment arms
    Fmax : (nbMuscles,) maximal isometric forces
    """
    q = np.asarray(q).reshape(-1)

    nb_q = model.nbQ()
    nb_mus = model.nbMuscles()

    # ---- Muscle length Jacobian ----
    # Shape: (nbMuscles, nbQ)
    J = model.musclesLengthJacobian(q).to_array()

    if J.shape != (nb_mus, nb_q):
        raise RuntimeError(
            f"Unexpected Jacobian shape {J.shape}, "
            f"expected ({nb_mus}, {nb_q})"
        )

    # ---- Moment arms ----
    R = -J.T  # (nbQ, nbMuscles)

    # ---- Max isometric forces ----
    Fmax = np.array(
        [model.muscle(i).characteristics().forceIsoMax()
         for i in range(nb_mus)],
        dtype=float,
    )

    return R, Fmax




def compute_muscle_forces_from_activation(model: biorbd.Model, q: np.ndarray, qdot: np.ndarray, a: np.ndarray):
    """
    Compute muscle forces (nbMuscles,) for a single frame, using biorbd muscleForces.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    nb_mus = model.nbMuscles()
    if a.shape[0] != nb_mus:
        raise ValueError(f"Activation size {a.shape[0]} != nbMuscles {nb_mus}")

    states = model.stateSet()
    for i in range(nb_mus):
        states[i].setActivation(float(a[i]))

    forces = model.muscleForces(states, q, qdot).to_array()
    forces = np.asarray(forces).reshape(-1)
    return forces


# ----------------------------
# Static Optimization (QP) per frame
# ----------------------------
def make_so_solver(nb_mus: int, nb_tau: int, qp_solver_name=QP_SOLVER):
    """
    Build a reusable QP solver in CasADi:
      min  || A a - tau ||^2 + w_act ||a||^2 + w_emg ||a - emg||^2
      s.t. 0 <= a <= 1
    We will provide A, tau, emg, weights as parameters each call.
    """
    a = ca.MX.sym("a", nb_mus)

    # parameters
    A = ca.MX.sym("A", nb_tau, nb_mus)     # map activations -> torques
    tau = ca.MX.sym("tau", nb_tau)         # target torques
    emg = ca.MX.sym("emg", nb_mus)         # desired activations (from EMG mapping)
    w_tau = ca.MX.sym("w_tau", 1)
    w_act = ca.MX.sym("w_act", 1)
    w_emg = ca.MX.sym("w_emg", 1)

    tau_m = A @ a

    cost = w_tau * ca.sumsqr(tau_m - tau) + w_act * ca.sumsqr(a) + w_emg * ca.sumsqr(a - emg)

    qp = {"x": a, "f": cost, "g": ca.MX()}  # no extra constraints (bounds only)

    try:
        solver = ca.qpsol("so_qp", qp_solver_name, qp)
    except Exception:
        # fallback
        solver = ca.qpsol("so_qp", "osqp", qp)

    return solver, (A, tau, emg, w_tau, w_act, w_emg), a


def solve_so_frame(
    solver,
    params_syms,
    A_np: np.ndarray,
    tau_np: np.ndarray,
    emg_np: np.ndarray,
    w_tau_val: float,
    w_act_val: float,
    w_emg_val: float,
):
    A_sym, tau_sym, emg_sym, w_tau_sym, w_act_sym, w_emg_sym = params_syms

    nb_mus = emg_np.shape[0]

    # bounds
    lbx = np.zeros(nb_mus)
    ubx = np.ones(nb_mus)

    # casadi expects dict with p for parameters in order:
    # We'll stack parameters manually using "p" only if solver supports it,
    # but easier is to pass via "lam_x0"? no. We'll use "p" by creating "p" vector.
    # Simpler: build a function that maps params to solution each call is heavier.
    #
    # Better: use solver with named arguments:
    arg = {
        "lbx": lbx,
        "ubx": ubx,
        "p": ca.vertcat(
            ca.reshape(A_np, -1, 1),
            tau_np.reshape(-1, 1),
            emg_np.reshape(-1, 1),
            np.array([[w_tau_val]], dtype=float),
            np.array([[w_act_val]], dtype=float),
            np.array([[w_emg_val]], dtype=float),
        )
    }

    # BUT this requires qp to have "p". To do that we must define qp["p"].
    # Therefore, we instead provide a wrapper that includes p explicitly.
    raise RuntimeError("Internal: solver built without parameter vector. Use build_so_solver_with_p().")


def build_so_solver_with_p(nb_mus: int, nb_tau: int, qp_solver_name=QP_SOLVER):
    """
    Same as make_so_solver but with single parameter vector p to feed numeric values.
    """
    a = ca.MX.sym("a", nb_mus)

    # parameter vector layout:
    # p = [vec(A) ; tau ; emg ; w_tau ; w_act ; w_emg]
    p = ca.MX.sym("p", nb_tau * nb_mus + nb_tau + nb_mus + 3)

    off = 0
    A_vec = p[off: off + nb_tau * nb_mus]
    off += nb_tau * nb_mus
    tau = p[off: off + nb_tau]
    off += nb_tau
    emg = p[off: off + nb_mus]
    off += nb_mus
    w_tau = p[off]
    w_act = p[off + 1]
    w_emg = p[off + 2]

    A = ca.reshape(A_vec, nb_tau, nb_mus)
    tau_m = A @ a

    cost = w_tau * ca.sumsqr(tau_m - tau) + w_act * ca.sumsqr(a) + w_emg * ca.sumsqr(a - emg)

    qp = {"x": a, "f": cost, "g": ca.MX(), "p": p}

    try:
        solver = ca.qpsol("so_qp", qp_solver_name, qp)
    except Exception:
        solver = ca.qpsol("so_qp", "osqp", qp)

    return solver


def run_so_frame(
    solver,
    A_np: np.ndarray,
    tau_np: np.ndarray,
    emg_np: np.ndarray,
    w_tau_val: float,
    w_act_val: float,
    w_emg_val: float,
):
    nb_tau, nb_mus = A_np.shape
    if tau_np.shape[0] != nb_tau:
        raise ValueError(f"tau size {tau_np.shape[0]} != nb_tau {nb_tau}")
    if emg_np.shape[0] != nb_mus:
        raise ValueError(f"emg size {emg_np.shape[0]} != nb_mus {nb_mus}")

    p = np.concatenate([
        A_np.reshape(-1),
        tau_np.reshape(-1),
        emg_np.reshape(-1),
        np.array([w_tau_val, w_act_val, w_emg_val], dtype=float),
    ])

    lbx = np.zeros(nb_mus)
    ubx = np.ones(nb_mus)

    sol = solver(lbx=lbx, ubx=ubx, p=p)
    a_opt = np.array(sol["x"]).reshape(-1)

    return np.clip(a_opt, EPS_ACT, 1.0 - EPS_ACT)


# ----------------------------
# Main
# ----------------------------
def main():
    model = biorbd.Model(MODEL_PATH)

    m = model.muscle(0).characteristics()
    print(dir(m))

    nbQ = model.nbQ()
    nbTau = model.nbGeneralizedTorque()
    nbMus = model.nbMuscles()

    print("Model loaded.")
    print(f"  nbQ   = {nbQ}")
    print(f"  nbTau = {nbTau}")
    print(f"  nbMus = {nbMus}")

    # Load data
    q = np.load(Q_PATH)
    qdot = np.load(QDOT_PATH)
    tau = np.load(TAU_PATH)
    emg_env = np.load(EMG_PATH)

    # EMG normalisé entre 0 et 100  →  0 à 1
    emg_env = emg_env / 100.0

    emg_env = np.clip(emg_env, 0.0, 1.0)

    # Fix shapes
    q = _maybe_transpose_to_dof_by_frames(q, nbQ, "q")
    qdot = _maybe_transpose_to_dof_by_frames(qdot, nbQ, "qdot")
    tau = _maybe_transpose_frames_match(tau, q.shape[1], "tau")
    emg_env = _maybe_transpose_frames_match(emg_env, q.shape[1], "emg_env")

    # Windowing
    n_frames_total = q.shape[1]
    if not (0 <= FIRST < END <= n_frames_total):
        raise ValueError(f"Invalid window FIRST={FIRST}, END={END}, with nFrames={n_frames_total}")

    q = q[:, FIRST:END]
    qdot = qdot[:, FIRST:END]
    tau = tau[:, FIRST:END]
    emg_env = emg_env[:, FIRST:END]

    n_frames = q.shape[1]
    print(f"Data windowed: nFrames={n_frames}")

    # Build EMG mapping
    track_idx, emg_src_idx, tracked_muscle_names = build_emg_to_muscle_mapping(
        model=model,
        emg_to_muscle_dict=emg_to_muscle,
        verbose=True,
    )

    # IMPORTANT: your emg_env has only 10 channels (0..9) here, so indices must be valid
    if np.max(emg_src_idx) >= emg_env.shape[0]:
        raise ValueError(
            f"emg_env has {emg_env.shape[0]} channels but mapping asks for channel {np.max(emg_src_idx)}.\n"
            f"Fix emg_to_muscle keys or emg_env channel order."
        )

    nbTrackedMus = track_idx.shape[0]
    print(f"\nTracked muscles (after expansion): {nbTrackedMus}")

    # Build SO solver for tracked muscles only
    solver = build_so_solver_with_p(nbTrackedMus, nbTau, qp_solver_name=QP_SOLVER)

    # Prepare outputs
    times = {}
    dic_to_save = {
        "mus_act": np.zeros((nbTrackedMus, n_frames)),
        "res_tau": np.zeros((nbTau, n_frames)),
        "mus_force": np.zeros((nbMus, n_frames)),
        "emg_proc": np.zeros((nbTrackedMus, n_frames)),
        "tracked_muscle_names": tracked_muscle_names,  # list
        "track_idx": track_idx,                        # ndarray
    }

    # Si modèle avec base flottante (souvent les 6 premiers ddl)
    if model.nbQ() == 18:
        q[:6, :] = 0.0
        qdot[:6, :] = 0.0
        tau[:6, :] = 0.0

    # Loop frames
    t0 = time.time()
    for k in range(n_frames):
        qk = q[:, k].reshape(-1)
        qdotk = qdot[:, k].reshape(-1)
        tauk = tau[:, k].reshape(-1)

        # Build A for tracked muscles only: A = R[:, track_idx] * diag(Fmax[track_idx])
        R, Fmax = get_moment_arms_and_fmax(model, qk)

        # Select tracked muscles columns
        R_tr = R[:, track_idx]                # (nbTau x nbTrackedMus)
        F_tr = Fmax[track_idx]                # (nbTrackedMus,)

        # Linear map activations -> torques
        A = R_tr * F_tr.reshape(1, -1)        # broadcast -> (nbTau x nbTrackedMus)

        # Build EMG vector for tracked muscles by duplicating corresponding channel
        emgk_src = emg_env[:, k].reshape(-1)  # (nbEmgChannels,)
        emgk_tr = emgk_src[emg_src_idx]       # (nbTrackedMus,)

        # Solve SO for activations of tracked muscles
        tic = time.time()
        a_tr = run_so_frame(
            solver=solver,
            A_np=A,
            tau_np=tauk,
            emg_np=emgk_tr,
            w_tau_val=W_TAU,
            w_act_val=W_ACT,
            w_emg_val=W_EMG,
        )
        times["so_last"] = time.time() - tic

        # Residual torques (tracking error)
        tau_hat = A @ a_tr
        res_tau = tau_hat - tauk

        # Rebuild full activation vector (nbMus) with zeros elsewhere
        a_full = np.zeros((nbMus,), dtype=float)
        a_full[track_idx] = a_tr

        # Compute muscle forces (full set)
        f_full = compute_muscle_forces_from_activation(model, qk, qdotk, a_full)

        # Save
        dic_to_save["mus_act"][:, k] = a_tr
        dic_to_save["res_tau"][:, k] = res_tau
        dic_to_save["mus_force"][:, k] = f_full
        dic_to_save["emg_proc"][:, k] = emgk_tr

        if (k + 1) % 200 == 0:
            print(f"Frame {k+1}/{n_frames} done.")

    times["total"] = time.time() - t0

    print("\nDone.")
    print(f"Total time: {times['total']:.3f} s")
    print("mus_act:", dic_to_save["mus_act"].shape)
    print("mus_force:", dic_to_save["mus_force"].shape)
    print("res_tau:", dic_to_save["res_tau"].shape)
    print("emg_proc:", dic_to_save["emg_proc"].shape)

    # Example: save results if you want
    # np.save("so_mus_act.npy", dic_to_save["mus_act"])
    # np.save("so_mus_force.npy", dic_to_save["mus_force"])
    # np.save("so_res_tau.npy", dic_to_save["res_tau"])
    # np.save("so_emg_proc.npy", dic_to_save["emg_proc"])


    res_tau = dic_to_save["res_tau"]  # (18, nFrames)

    # moyenne de l'erreur absolue par DoF
    err_mean = np.mean(np.abs(res_tau), axis=1)

    print("Mean |res_tau| per DoF:")
    for i, v in enumerate(err_mean):
        print(f"DoF {i:2d}: {v:.3f}")

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(res_tau.shape[0]), err_mean)
    plt.title("Moyenne de |res_tau| par DoF")
    plt.xlabel("DoF index")
    plt.ylabel("mean |res_tau|")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

    mus_act = dic_to_save["mus_act"]  # (17, nFrames)

    plt.figure(figsize=(12, 6))
    for i in range(mus_act.shape[0]):
        plt.plot(mus_act[i, :], label=f"Muscle {i}")

    plt.title("Activations musculaires (Static Optimization)")
    plt.xlabel("Frame")
    plt.ylabel("Activation")
    #plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()



    return dic_to_save, times


if __name__ == "__main__":
    main()
