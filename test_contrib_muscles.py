import sys
import time
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import biorbd

# -----------------------------
# Shared state message
# -----------------------------
@dataclass
class Params:
    muscle_idx: int
    dof_idx: int
    activation: float


# -----------------------------
# Viewer process (pyorerun + rerun + compute)
# -----------------------------

def eigen_to_np(x):
    # cas le plus fréquent dans biorbd python
    if hasattr(x, "to_array"):
        return np.array(x.to_array())
    # parfois c'est directement convertible via np.asarray
    try:
        return np.asarray(x)
    except Exception:
        pass
    # dernier recours: construire depuis listes
    if hasattr(x, "__len__"):
        return np.array(x)
    raise TypeError(f"Impossible de convertir en numpy: type={type(x)}")

def to_numeric_scalar(x):
    """Convertit x (float/int/numpy/biorbd wrapper) en float Python, de façon robuste."""
    # déjà numérique
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)

    # wrappers biorbd/Eigen : souvent to_array()
    if hasattr(x, "to_array"):
        arr = np.array(x.to_array())
        arr = arr.reshape((-1,))
        if arr.size == 0:
            raise ValueError("to_array() vide")
        return to_numeric_scalar(arr[0]) if arr.dtype == object else float(arr[0])

    # numpy array
    if isinstance(x, np.ndarray):
        x = x.reshape((-1,))
        if x.size == 0:
            raise ValueError("ndarray vide")
        return to_numeric_scalar(x[0]) if x.dtype == object else float(x[0])

    # séquences (tuple/list)
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("liste/tuple vide")
        return to_numeric_scalar(x[0])

    # dernier recours: certains wrappers supportent float()
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"Impossible de convertir en float: type={type(x)} value={x}") from e


def torque_component(tau_vec, dof_idx: int) -> float:
    """
    Extrait tau[dof_idx] en float, que tau_vec soit:
    - un wrapper GeneralizedTorque vectoriel
    - un tuple/list de composants
    - un wrapper scalaire
    """
    # si c'est une séquence directe
    if isinstance(tau_vec, (list, tuple)):
        return to_numeric_scalar(tau_vec[dof_idx])

    # cas wrapper vectoriel: on tente d'indexer
    try:
        return to_numeric_scalar(tau_vec[dof_idx])
    except Exception:
        pass

    # cas wrapper vectoriel: on tente to_array global
    if hasattr(tau_vec, "to_array"):
        arr = np.array(tau_vec.to_array()).reshape((-1,))
        if arr.dtype == object:
            return to_numeric_scalar(arr[dof_idx])
        return float(arr[dof_idx])

    # sinon c'est peut-être déjà un scalaire
    return to_numeric_scalar(tau_vec)



def viewer_process(model_path: str, q_params: mp.Queue):
    import threading, time
    import numpy as np
    import biorbd
    import rerun as rr
    from pyorerun import LiveModelAnimation

    model = biorbd.Model(model_path)
    nb_q = model.nbQ()
    qdot = np.zeros((nb_q,))

    rr.init("muscle_torque_explorer", spawn=False)

    animation = LiveModelAnimation(model_path, with_q_charts=False)
    print("with_q_charts =", animation.with_q_charts if hasattr(animation, "with_q_charts") else "unknown")

    # --- states musculaires robustes (cf doc biorbd) ---
    nb_mus = model.nbMuscleTotal() if hasattr(model, "nbMuscleTotal") else model.nbMuscles()

    def make_states(model, one_active_idx: int, activation: float):
        nb_mus = model.nbMuscles()
        states = []
        for i in range(nb_mus):
            a = activation if i == one_active_idx else 0.0
            states.append(biorbd.State(a, a))  # (excitation, activation)
        return states

    # récupérer l'objet muscles (parfois tuple)
    muscles_obj = model.muscles()
    if isinstance(muscles_obj, tuple):
        muscles_obj = next((x for x in muscles_obj if hasattr(x, "muscularJointTorque")), None)

    # paramètres courants
    cur_muscle, cur_dof, cur_a = 0, 0, 0.3

    def rr_set_frame(frame: int):
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("frame", frame)

    def get_q1d():
        # essaie plusieurs attributs
        for attr in ("q", "q_current", "current_q", "_q"):
            if hasattr(animation, attr):
                v = getattr(animation, attr)
                if isinstance(v, np.ndarray) and (v.shape == (nb_q,) or v.shape == (nb_q, 1)):
                    return np.asarray(v).reshape((-1,))
        return np.zeros((nb_q,))

    def to_np(x):
        if hasattr(x, "to_array"):
            return np.array(x.to_array())
        return np.asarray(x)

    def worker():
        nonlocal cur_muscle, cur_dof, cur_a
        frame = 0
        while True:
            # paramètres UI (non bloquant)
            try:
                while True:
                    p = q_params.get_nowait()
                    cur_muscle, cur_dof, cur_a = p.muscle_idx, p.dof_idx, p.activation
            except Exception:
                pass

            q = get_q1d()

            # moment arm via Jacobien des longueurs: r = -dL/dq :contentReference[oaicite:2]{index=2}
            J = to_np(model.musclesLengthJacobian(q))
            r = -float(J[cur_muscle, cur_dof])

            # couple musculaire en activant un seul muscle
            states = make_states(model, cur_muscle, cur_a)

            # selon version: via model ou via muscles_obj
            if hasattr(model, "muscularJointTorque"):
                tau_vec = model.muscularJointTorque(states, q, qdot)
            else:
                tau_vec = muscles_obj.muscularJointTorque(states, q, qdot)

            tau_arr = to_np(tau_vec).reshape((-1,))
            tau = float(tau_arr[cur_dof])

            rr_set_frame(frame)
            rr.log("analysis/moment_arm", rr.Scalar(r))
            rr.log("analysis/tau", rr.Scalar(tau))
            rr.log("analysis/q_selected", rr.Scalar(float(q[cur_dof])))
            rr.log("analysis/activation", rr.Scalar(cur_a))

            frame += 1
            time.sleep(0.03)

    threading.Thread(target=worker, daemon=True).start()

    # IMPORTANT: affiche le modèle + sliders
    animation.rerun()



# -----------------------------
# UI process (PyQt5)
# -----------------------------
def pick_biomod_file() -> str | None:
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Choisir un modèle .bioMod",
        "",
        "Biorbd models (*.bioMod);;All files (*)",
    )
    return path if path else None


def ui_main(model_path: str):
    from PyQt5 import QtWidgets, QtCore

    model = biorbd.Model(model_path)
    muscle_names = [model.muscle(i).name().to_string() for i in range(model.nbMuscles())]
    dof_names = [f"DoF {i}" for i in range(model.nbGeneralizedTorque())]

    q_params = mp.Queue()

    # lance le viewer dans un process séparé (stable sur macOS)
    p = mp.Process(target=viewer_process, args=(model_path, q_params), daemon=True)
    p.start()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QWidget()
    w.setWindowTitle("Muscle torque explorer")

    muscle_cb = QtWidgets.QComboBox()
    muscle_cb.addItems(muscle_names)

    dof_cb = QtWidgets.QComboBox()
    dof_cb.addItems(dof_names)

    act = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    act.setRange(0, 100)
    act.setValue(30)
    act_lbl = QtWidgets.QLabel("0.30")

    def push():
        a = act.value() / 100.0
        act_lbl.setText(f"{a:.2f}")
        q_params.put(Params(muscle_cb.currentIndex(), dof_cb.currentIndex(), a))

    muscle_cb.currentIndexChanged.connect(push)
    dof_cb.currentIndexChanged.connect(push)
    act.valueChanged.connect(push)

    form = QtWidgets.QFormLayout(w)
    form.addRow("Muscle", muscle_cb)
    form.addRow("DoF", dof_cb)

    row = QtWidgets.QHBoxLayout()
    row.addWidget(act, 1)
    row.addWidget(act_lbl)
    form.addRow("Activation", row)

    push()
    w.show()
    app.exec()

    # fermeture
    if p.is_alive():
        p.terminate()
        p.join(timeout=1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # important macOS

    model_path = sys.argv[1] if len(sys.argv) >= 2 else None
    if not model_path:
        model_path = pick_biomod_file()
        if not model_path:
            print("Aucun fichier sélectionné. Fin.")
            raise SystemExit(0)

    ui_main(model_path)
