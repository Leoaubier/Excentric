from pyomeca import Analogs
import matplotlib.pyplot as plt
import pandas as pd


def clean_sto(path, output_path):
    # Lire avec pandas en auto-détectant le header
    df = pd.read_csv(path, delim_whitespace=True, comment='#')

    # Forcer la colonne time à être numérique
    df['time'] = pd.to_numeric(df['time'], errors='coerce')

    # Sauver un fichier sto propre
    df.to_csv(output_path, sep="\t", index=False)

clean_sto(
    "/Users/leo/Desktop/Projet/modele_opensim/inverse_dynamics.sto",
    "/Users/leo/Desktop/Projet/modele_opensim/inverse_dynamics_clean.sto"
)

# --- 1. Lecture du fichier .sto ---
# Pyomeca lit les fichiers type C3D/TRC/STO comme des tableaux xarray
data = Analogs.from_sto("/Users/leo/Desktop/Projet/modele_opensim/inverse_dynamics_clean.sto")

# Pour voir les dimensions et les coordonnées (labels, temps, etc.)
print(data)
print("Dimensions :", data.dims)
print("Coords :", data.coords)

# --- 2. Accéder à un canal (une colonne) particulier ---
# Par exemple, si ton fichier contient un signal "GRF_X"
grf_x = data.sel(channel="GRF_X")

print(grf_x.values)   # valeurs numériques
print(grf_x.time)     # vecteur temps

# --- 3. Tracer un signal ---
plt.figure()
plt.plot(data.time, grf_x.values.T)  # .T pour s’assurer de la bonne orientation
plt.xlabel("Temps")
plt.ylabel("GRF_X")
plt.title("Force de réaction au sol - X")
plt.show()

