import numpy as np
import biorbd
import matplotlib.pyplot as plt

# --------------------------------------------------------------
#  Test 1 : COHÉRENCE MATHÉMATIQUE
# --------------------------------------------------------------
def RT_from_numpy(R, t=None):
    if t is None:
        t = np.zeros(3)

    rot = biorbd.Rotation(
        float(R[0,0]), float(R[0,1]), float(R[0,2]),
        float(R[1,0]), float(R[1,1]), float(R[1,2]),
        float(R[2,0]), float(R[2,1]), float(R[2,2])
    )

    trans = biorbd.Vector3d(float(t[0]), float(t[1]), float(t[2]))

    return biorbd.RotoTrans(rot, trans)

def test_math_consistency():
    print("----- TEST 1 : cohérence mathématique -----")

    # Random rotation matrix
    def random_rot():
        th = np.random.rand(3) * 2*np.pi
        Rx = np.array([[1,0,0],[0,np.cos(th[0]),-np.sin(th[0])],[0,np.sin(th[0]),np.cos(th[0])]])
        Ry = np.array([[np.cos(th[1]),0,np.sin(th[1])],[0,1,0],[-np.sin(th[1]),0,np.cos(th[1])]])
        Rz = np.array([[np.cos(th[2]),-np.sin(th[2]),0],[np.sin(th[2]),np.cos(th[2]),0],[0,0,1]])
        return Rz @ Ry @ Rx

    R = random_rot()
    # Convertir la matrice numpy en rotation biorbd
    rt = RT_from_numpy(R)

    e = biorbd.RotoTrans.toEulerAngles(rt, seq= "xyz")
    rt2 = biorbd.RotoTrans.fromEulerAngles(e,seq="xyz",trans = biorbd.Vector3d(0, 0, 0))

    error = np.linalg.norm(rt.to_array() - rt2.to_array())
    print("Erreur reconstruction RT->Euler->RT = ", error)
    return error


# --------------------------------------------------------------
#  Test 2 : COHÉRENCE BIOMÉCANIQUE – épaule plane–elevation–rotation
# --------------------------------------------------------------

def rotation_x(a): return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
def rotation_y(a): return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
def rotation_z(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])

def test_biomechanical():
    print("\n----- TEST 2 : cohérence biomécanique épaule -----")

    plane     = np.deg2rad(30)
    elevation = np.deg2rad(60)
    rotation  = np.deg2rad(20)

    # Conventions ISB : plane (Y), elevation (X), rotation (Z)
    R = rotation_z(rotation) @ rotation_x(elevation) @ rotation_y(plane)
    rt = RT_from_numpy(R)

    e = biorbd.RotoTrans.toEulerAngles(rt, seq="xyz")
    rt2 = biorbd.RotoTrans.fromEulerAngles(e,seq="xyz",trans = biorbd.Vector3d(0, 0, 0))

    print("Angles extraits (rad) = ", e)
    print("Erreur RT original vs reconstruit : ", np.linalg.norm(rt.to_array()-rt2.to_array()))
    return e


# --------------------------------------------------------------
#  Test 3 : TEST SUR TON MODELE
# --------------------------------------------------------------

def test_model(model_path):
    print("\n----- TEST 3 : modèle Biorbd -----")

    model = biorbd.Model(model_path)

    # Exemple : posture aléatoire
    q = np.zeros(model.nbQ())
    q[:] = 0.3*np.random.randn(model.nbQ())

    # RT des segments
    rt_scap = model.globalJCS(model.getDofIndex("scapula_left"), q)
    rt_hum  = model.globalJCS(model.getDofIndex("humerus_left"), q)

    # RT relative humerus/scapula
    rt_rel = rt_scap.transpose() * rt_hum

    # extraction via biorbd
    ang = biorbd.RotoTrans.toEulerAngles(rt_rel, "xyz")
    return ang, rt_rel


# --------------------------------------------------------------
#  Test 4 : comparaison Q_recons vs Euler(RT_rel)
# --------------------------------------------------------------

def compare_curves(q_recons, euler_rel, labels):
    print("\n----- TEST 4 : comparaison courbes -----")

    N = q_recons.shape[1]
    t = np.linspace(0,1,N)

    plt.figure()
    for i in range(3):
        plt.plot(t, q_recons[i,:], label=f"Q_recons {labels[i]}")
        plt.plot(t, euler_rel[i,:], '--', label=f"Euler RT {labels[i]}")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("angle (rad)")
    plt.grid()
    plt.show()

    rms = np.sqrt(np.mean((q_recons - euler_rel)**2))
    print("Erreur RMS = ", rms)


# --------------------------------------------------------------
# EXÉCUTION GLOBALE
# --------------------------------------------------------------

if __name__ == "__main__":

    # 1. Test mathématique
    test_math_consistency()

    # 2. Test biomécanique
    test_biomechanical()

    # 3. Test sur le modèle
    ang, rt_rel = test_model("/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod")
    print("\nAngles extraits du modèle :", ang)

    # 4. Exemple de comparaison (à remplacer par tes vrais signaux)
    N = 200
    q_recons_fake = np.vstack([
        np.sin(np.linspace(0,2*np.pi,N)),
        np.cos(np.linspace(0,2*np.pi,N)),
        0.5*np.sin(2*np.linspace(0,2*np.pi,N))
    ])
    euler_fake = q_recons_fake * 0.9

    compare_curves(q_recons_fake, euler_fake, ["plane", "elev", "rot"])
