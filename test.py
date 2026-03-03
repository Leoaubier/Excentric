import numpy as np
import matplotlib.pyplot as plt

# =========================
# Paramètres
# =========================
PUISSANCE = 1000
n_cycles = 4
points_per_cycle = 500
T = 1.0                     # durée d’un cycle
phase_shift = np.pi      # décalage temporel (25% du cycle)

# =========================
# Temps
# =========================
t = np.linspace(0, n_cycles * T, n_cycles * points_per_cycle)

# =========================
# Signal 1 : rampe 0 → 2π (retour vertical)
# =========================
triangle_1 = (t % T) / T * 2 * np.pi

# =========================
# Signal 2 : sinus
# =========================
sin_1 = np.sin(2 * np.pi * t / T)

# =========================
# Signal 3 : rampe décalée
# =========================
triangle_2 = ((t + phase_shift) % T) / T * 2 * np.pi

#==========================
# cos

cos_2 = np.cos(2 * np.pi * (t + np.pi) / T)

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 6))

plt.plot(t, triangle_1, label="Rampe 0→2π")
plt.plot(t, sin_1, label="Sinus")
plt.plot(t, triangle_2, label="Rampe décalée")
plt.plot(t, cos_2, label="Cosinus")


plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.title("Signaux périodiques")
plt.legend()
plt.grid(True)

plt.show()

np.save(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/emg_processed_resampled.npy", (sin_1,sin_1,sin_1,sin_1,sin_1,sin_1,sin_1,sin_1,sin_1,sin_1,sin_1))
np.save(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/crank_angle.npy", triangle_1)
np.save(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/emg_processed_resampled.npy", (cos_2,cos_2,cos_2,cos_2,cos_2,cos_2,cos_2,cos_2,cos_2,cos_2,cos_2))
np.save(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/crank_angle.npy", triangle_2)