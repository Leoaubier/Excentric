import numpy as np
import biorbd
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER INPUTS
# ============================================================
PLOT = False

ESSAI = "Collecte_18_03"
MODE_PEDALAGE = "concentric"
PUISSANCE = "40"

MODEL_PATH = f"/Users/leo/Desktop/Projet/{ESSAI}/model_{ESSAI}.bioMod"
Q_PATH     = f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/q_inverse_kinematic.npy"

FIRST_FRAME = 2000
LAST_FRAME  = 3400                     # None = jusqu'à la fin

OUT_CSV = f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/moment_arms_by_dof.csv"                         # ex: "moment_arms_by_dof.csv" ou None
# ============================================================


def load_q(q_path: str, n_q: int) -> np.ndarray:
    q = np.load(q_path)
    q = np.asarray(q, dtype=float)

    if q.ndim != 2:
        raise ValueError(f"q must be 2D, got shape {q.shape}")

    if q.shape[0] == n_q:
        return q
    if q.shape[1] == n_q:
        return q.T

    raise ValueError(
        f"q has shape {q.shape} but model expects nQ={n_q}. "
        "Expected (nQ,nFrames) or (nFrames,nQ)."
    )


def compute_moment_arms_stats_by_dof(model: biorbd.Model, q: np.ndarray, f0: int, f1: int):
    """
    Retourne un DataFrame indexé par muscle, avec colonnes MultiIndex:
      (dof_name, 'min'/'mean'/'max')
    """
    n_q = model.nbQ()
    n_mus = model.nbMuscles()

    dof_names = [model.nameDof()[i].to_string() for i in range(n_q)]
    mus_names = [model.muscle(i).name().to_string() for i in range(n_mus)]

    f0 = int(max(0, f0))
    f1 = q.shape[1] if f1 is None else int(min(q.shape[1], f1))
    if f1 <= f0:
        raise ValueError(f"Invalid frame range: FIRST_FRAME={f0}, LAST_FRAME={f1}")

    n_frames = f1 - f0

    # accumulate: (nMus, nQ, nFrames)
    all_ma = np.zeros((n_mus, n_q, n_frames), dtype=float)

    for k, f in enumerate(range(f0, f1)):
        q_f = q[:, f]
        ma = model.musclesLengthJacobian(q_f).to_array()  # (nMus, nQ)
        all_ma[:, :, k] = ma

    ma_min  = np.min(all_ma, axis=2)   # (nMus, nQ)
    ma_mean = np.mean(all_ma, axis=2)
    ma_max  = np.max(all_ma, axis=2)

    # Build DataFrame: index=muscles, columns=(dof, stat)
    cols = pd.MultiIndex.from_product([dof_names, ["min", "mean", "max"]], names=["DoF", "stat"])
    data = np.zeros((n_mus, len(dof_names) * 3), dtype=float)

    for j, dof in enumerate(dof_names):
        data[:, 3*j + 0] = ma_min[:, j]
        data[:, 3*j + 1] = ma_mean[:, j]
        data[:, 3*j + 2] = ma_max[:, j]

    df = pd.DataFrame(data, index=mus_names, columns=cols)
    df.index.name = "muscle"

    return df, (f0, f1)


def main():
    model = biorbd.Model(MODEL_PATH)
    q = load_q(Q_PATH, model.nbQ())

    df, (f0, f1) = compute_moment_arms_stats_by_dof(model, q, FIRST_FRAME, LAST_FRAME)

    # Affiche un aperçu (pandas)
    print(f"\n=== Moment arms by DoF | frames [{f0}:{f1}] ===")
    print(df.round(6))

    # Export CSV si demandé (colonnes aplaties)
    if OUT_CSV is not None:
        df_out = df.copy()
        df_out.columns = [f"{dof}__{stat}" for (dof, stat) in df_out.columns]
        df_out.to_csv(OUT_CSV)
        print(f"\nSaved: {OUT_CSV}")

    # Exemple de filtre : ne garder qu'un DoF contenant 'Elbow'
    # df_elbow = df.loc[:, df.columns.get_level_values("DoF").str.contains("Elbow", case=False)]
    # print(df_elbow.round(6))


if __name__ == "__main__":
    main()