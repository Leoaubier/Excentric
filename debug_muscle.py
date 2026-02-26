import numpy as np
import biorbd
import matplotlib.pyplot as plt
import casadi as ca

Muscletoplot = 32
Firstdoftoplot = 14
# ============================================================
# USER INPUTS
# ============================================================
PLOT = False

MODE_PEDALAGE = "eccentric"
PUISSANCE = "40"

MODEL_PATH = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie_vtp.bioMod"
Q_PATH     = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/q_inverse_kinematic.npy"
QDOT_PATH     = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/qdot_inverse_kinematic.npy"
TAU_PATH  = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/tau_inverse_dynamic.npy"


FIRST_FRAME = 3000
LAST_FRAME  = 4000                    # None = jusqu'à la fin

# DoF filter: laisser None pour tout, ou mettre un mot-clé (ex: "Elbow")
DOF_CONTAINS = None  # ex: "Elbow" / "elbow" / "Shoulder" / etc.

# Plot options
PLOT_PER_DOF = True
SHOW_TOP_K_MUSCLES = 6   # dans le report, montre les muscles qui contribuent le plus à la capacité
# ============================================================


def get_lm_lopt(model_path, q, qdot):

    model = biorbd.Model(model_path)
    nMus = model.nbMuscles()
    nFrames = q.shape[1]

    lm = np.zeros((nMus, nFrames))
    lm2 = np.zeros((nMus, nFrames))
    A = np.zeros((nMus, nFrames))
    v_musc = np.zeros((nMus, nFrames))

    lopt = np.zeros(nMus)
    muscle_names = []

    # récupérer paramètres musculaires (constants)
    for i in range(nMus):
        mus = model.muscle(i)
        lopt[i] = mus.characteristics().optimalLength()
        muscle_names.append(mus.name().to_string())

    # boucle frames
    for k in range(nFrames):
        qk = q[:, k]
        qdotk = qdot[:, k]
        model.updateMuscles(qk)
        print(f"frame {k}")

        for i in range(nMus):
            mus = model.muscle(i)


            lm[i, k] = mus.length(model, qk, True)  # longueur de fibre musculaire
            lmt = mus.musculoTendonLength(model, qk, True)  # longueur muscle + tendon
            lts = mus.characteristics().tendonSlackLength() # longueur tendon
            penn = mus.characteristics().pennationAngle() # angle de pennation
            v_musc[i,k] = mus.velocity(model, qk,qdotk, True)
            lm2[i, k] = (lmt - lts) / np.cos(penn) # longueur musculaire estimée
            A[i,k] = lmt - lts


    lmtilde = lm / lopt[:, None]

    return lm, lopt, lmtilde, muscle_names , A, v_musc




def load_2d(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        x = np.load(path)
    elif path.endswith(".csv"):
        x = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported file: {path} (use .npy or .csv)")
    return np.asarray(x, dtype=float)


def orient_to_nq_nframes(x: np.ndarray, n_q: int, name="array") -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {x.shape}")
    if x.shape[0] == n_q:
        return x
    if x.shape[1] == n_q:
        return x.T
    raise ValueError(f"{name} has shape {x.shape}, expected (nQ,nFrames) or (nFrames,nQ) with nQ={n_q}")


def get_fmax_per_muscle(model: biorbd.Model) -> np.ndarray:
    """
    Retourne Fmax (N) pour chaque muscle.
    Selon versions biorbd, les getters peuvent varier -> on essaie plusieurs accès.
    """
    fmax = []
    for i in range(model.nbMuscles()):
        m = model.muscle(i)
        ch = m.characteristics()

        val = None
        # essais robustes
        for attr in ("forceIsoMax", "forceIsoMaximal", "forceIsoMaxValue"):
            if hasattr(ch, attr):
                cand = getattr(ch, attr)
                try:
                    val = cand() if callable(cand) else float(cand)
                except Exception:
                    pass
                if val is not None:
                    break

        if val is None:
            # fallback: certaines versions exposent directement sur muscle
            for attr in ("forceIsoMax", "maximalforce", "Fmax"):
                if hasattr(m, attr):
                    cand = getattr(m, attr)
                    try:
                        val = cand() if callable(cand) else float(cand)
                    except Exception:
                        pass
                    if val is not None:
                        break

        if val is None:
            raise RuntimeError(
                "Impossible de récupérer Fmax pour au moins un muscle. "
                "Dis-moi ta version biorbd (pip/conda) et je t’adapte le getter."
            )

        fmax.append(float(val))

    return np.array(fmax, dtype=float)


def main():
    model = biorbd.Model(MODEL_PATH)
    n_q = model.nbQ()
    n_mus = model.nbMuscles()

    dof_names = [model.nameDof()[i].to_string() for i in range(n_q)]
    mus_names = [model.muscle(i).name().to_string() for i in range(n_mus)]

    q = orient_to_nq_nframes(load_2d(Q_PATH)[:,FIRST_FRAME:LAST_FRAME], n_q, name="q")
    qdot = orient_to_nq_nframes(load_2d(QDOT_PATH)[:,FIRST_FRAME:LAST_FRAME], n_q, name="qdot")
    tau = orient_to_nq_nframes(load_2d(TAU_PATH)[:,FIRST_FRAME:LAST_FRAME], n_q, name="tau")


    n_frames = LAST_FRAME-FIRST_FRAME

    # DoF selection
    if DOF_CONTAINS is None:
        dof_idx = np.arange(n_q)
    else:
        mask = np.array([DOF_CONTAINS.lower() in dn.lower() for dn in dof_names])
        dof_idx = np.where(mask)[0]
        if dof_idx.size == 0:
            raise ValueError(f"No DoF matched DOF_CONTAINS='{DOF_CONTAINS}'. Available: {dof_names}")

    # Fmax
    Fmax = get_fmax_per_muscle(model)  # (nMus,)

    # store capacity envelopes
    cap_pos = np.zeros((n_q, n_frames), dtype=float)  # tau_max+ (>=0)
    cap_neg = np.zeros((n_q, n_frames), dtype=float)  # tau_max- (<=0)
    margin  = np.zeros((n_q, n_frames), dtype=float)  # signed margin: + means feasible, - means exceeding
    lmt_dot_from_J  = np.zeros((n_mus, n_frames), dtype=float)  # signed margin: + means feasible, - means exceeding
    Rtrilat = np.zeros((3,n_frames))
    # For reporting worst-case, keep the R*Fmax contributions per frame if needed
    # (on recompute for the worst frame only to stay light)

    for k, f in enumerate(range(n_frames)):
        q_f = q[:, f]
        qdot_f = qdot[:,f]
        R = model.musclesLengthJacobian(q_f).to_array()  # (nMus, nQ) = d(l_mt)/dq
        lmt_dot_from_J[:,f] = R @ qdot_f
        Rtrilat[0,f] = R[Muscletoplot,Firstdoftoplot]
        Rtrilat[1,f] = R[Muscletoplot,Firstdoftoplot+1]
        Rtrilat[2,f] = R[Muscletoplot,Firstdoftoplot+2]


        # muscle torque contribution bounds per DoF if fully activated at Fmax:
        # contrib = R_ij * Fmax_i
        contrib = (R.T * Fmax).T  # (nMus, nQ)

        # positive/negative capacity envelopes
        cap_pos[:, k] = np.sum(np.maximum(contrib, 0.0), axis=0)
        cap_neg[:, k] = np.sum(np.minimum(contrib, 0.0), axis=0)

        # compare to requested tau
        tau_f = tau[:, f]
        # margin defined so that positive => within capacity
        m = np.zeros(n_q, dtype=float)
        # if tau >=0: capacity is cap_pos; if tau <0: capacity is cap_neg
        pos = tau_f >= 0
        neg = ~pos
        m[pos] = cap_pos[pos, k] - tau_f[pos]
        m[neg] = tau_f[neg] - cap_neg[neg, k]  # since cap_neg is negative, tau - cap_neg >=0 is feasible
        margin[:, k] = m


    # Global worst case among selected DoF
    jg, kg = None, None
    sub = margin[dof_idx, :]
    if np.any(sub < 0):
        flat = np.argmin(sub)  # most negative
        jj = flat // sub.shape[1]
        kk = flat % sub.shape[1]
        jg = int(dof_idx[jj])
        kg = int(kk)
        f_worst = kg

        print("\n=== GLOBAL WORST (selected DoF) ===")
        print(f"DoF      : {dof_names[jg]}")
        print(f"Frame    : {f_worst}")
        print(f"tau      : {tau[jg, f_worst]: .3f}")
        print(f"cap_pos  : {cap_pos[jg, kg]: .3f}")
        print(f"cap_neg  : {cap_neg[jg, kg]: .3f}")
        print(f"margin   : {margin[jg, kg]: .3f}  (negative => exceeds)")

        # show configuration q at that frame
        q_w = q[:, f_worst]
        print("\nConfiguration q at worst frame:")
        for i in range(n_q):
            print(f"  {dof_names[i]:30s}  {q_w[i]: .6f}")

        # recompute contributions for that exact frame and list top muscles driving capacity in required sign
        Rw = model.musclesLengthJacobian(q_w).to_array()
        contrib_w = (Rw.T * Fmax).T  # (nMus,nQ)

        tau_w = tau[jg, f_worst]
        if tau_w >= 0:
            relevant = np.maximum(contrib_w[:, jg], 0.0)
            direction = "positive"
        else:
            relevant = np.minimum(contrib_w[:, jg], 0.0)
            relevant = -relevant  # magnitudes for ranking
            direction = "negative"

        top = np.argsort(relevant)[::-1][:SHOW_TOP_K_MUSCLES]
        print(f"\nTop {SHOW_TOP_K_MUSCLES} muscle contributors to {direction} capacity for {dof_names[jg]}:")
        for idx in top:
            val = contrib_w[idx, jg]
            print(f"  {mus_names[idx]:25s}  contrib={val: .6f}  (Fmax={Fmax[idx]:.1f})")

    else:
        print("\n✅ No exceedance detected on the selected DoF(s) over the analyzed frames.")

    # ============================================================
    # PLOTS
    # ============================================================

    lm, lopt, lmtilde, names, A, v_musc = get_lm_lopt(MODEL_PATH, q, qdot)

    plt.plot(Rtrilat[0,:]*1000, label=f"{model.muscleNames()[Muscletoplot].to_string()},{model.nameDof()[Firstdoftoplot].to_string()}")
    plt.plot(Rtrilat[1,:]*1000, label=f"{model.muscleNames()[Muscletoplot].to_string()},{model.nameDof()[Firstdoftoplot+1].to_string()}")
    plt.plot(Rtrilat[2,:]*1000, label=f"{model.muscleNames()[Muscletoplot].to_string()},{model.nameDof()[Firstdoftoplot+2].to_string()}")
    plt.plot(tau[Firstdoftoplot,:],label=f"{model.nameDof()[Firstdoftoplot].to_string()}")
    plt.plot(tau[Firstdoftoplot+1,:],label=f"{model.nameDof()[Firstdoftoplot+1].to_string()}")
    plt.plot(tau[Firstdoftoplot+2,:],label=f"{model.nameDof()[Firstdoftoplot+2].to_string()}")

    plt.legend()
    plt.show()

    for i in range(model.nbDof()):
        print(f"{model.nameDof()[i].to_string()} : min qdot : {min(qdot[i,:])}, max v_musc-lmt : {max(qdot[i,:])}")


    for i in range(model.nbMuscles()):
        print(f"{model.muscleNames()[i].to_string()} : min v_musc : {min(v_musc[i,:])}, max v_musc : {max(v_musc[i,:])}")

    for i in range(model.nbMuscles()):
        print(f"{model.muscleNames()[i].to_string()} : min Lm/Lopt : {min(lmtilde[i,:])}, max Lm/Lopt : {max(lmtilde[i,:])}")
        print(f"{model.muscleNames()[i].to_string()} : min Lmt-Lts : {min(A[i,:])}")

    for i in range(model.nbMuscles()):
        print(f"{model.muscleNames()[i].to_string()} : min Lm/Lopt : {min(R[i, :])}, mean R : {np.mean(R[i, :])}, max Lm/Lopt : {max(R[i, :])}")

    if PLOT == True:
        for i in range(model.nbMuscles()):
            plt.plot(lmtilde[i,:])
            plt.title(f"{model.muscleNames()[i].to_string()}")
            plt.xlabel("(lmt-lts)/cos(penn)")
            plt.ylabel("frame")
            plt.show()

        if PLOT_PER_DOF:
            t = np.arange(0, n_frames)

            for j in dof_idx:
                plt.figure(figsize=(10, 4))
                # capacity envelope
                plt.fill_between(t, cap_neg[j, :], cap_pos[j, :], alpha=0.2, label="muscle capacity envelope")
                # requested tau
                plt.plot(t, tau[j, :], linewidth=2, label="tau (requested)")



                plt.axhline(0, linewidth=1)
                plt.title(f"{dof_names[j]} | tau vs muscle capacity (frames {FIRST_FRAME}:{LAST_FRAME})")
                plt.xlabel("Frame")
                plt.ylabel("Torque [N.m] (or consistent units)")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()




if __name__ == "__main__":
    main()