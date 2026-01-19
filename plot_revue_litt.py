import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# Muscles communs
# =========================
muscles = [
    "Triceps brachii",
    "Biceps brachii",
    "Deltoïde antérieur",
    "Deltoïde postérieur",
    "Pectoralis major"
]

# =========================
# Activations normalisées (% cycle)
# =========================
study1 = {
    "Triceps brachii": (215, 30),
    "Biceps brachii": (100, 225),
    "Deltoïde antérieur": (120, 255),
    "Deltoïde postérieur": (265, 160),
    "Pectoralis major": (210, 300),
}
#study2 = {
#    "Biceps brachii": (175, 20),
#    "Deltoïde antérieur": (275, 85),
#    "Deltoïde postérieur": (95, 320),
#    "Pectoralis major": (300, 120),
#}

study2 = {
    "Triceps brachii": (210, 70),
    "Biceps brachii": (85, 290),
    "Deltoïde antérieur": (185, 355),
    "Deltoïde postérieur": (5, 230),
    "Pectoralis major": (210, 30),
}

# =========================
# Fonctions
# =========================
def deg_to_rad(deg):
    return np.deg2rad(deg)

def angular_segment(start_deg, end_deg, n=300):
    if end_deg >= start_deg:
        theta = np.linspace(start_deg, end_deg, n)
    else:
        theta = np.concatenate([
            np.linspace(start_deg, 360, n // 2),
            np.linspace(0, end_deg, n // 2)
        ])
    return deg_to_rad(theta)

# =========================
# Figure polaire
# =========================
fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"projection": "polar"})

ax.set_theta_zero_location("S")
ax.set_theta_direction(1)
ax.set_yticks([])

ax.set_thetagrids(
    np.arange(0, 360, 45),
    labels=[f"{d}°" for d in np.arange(0, 360, 45)]
)

# =========================
# Paramètres visuels
# =========================
base_radius = 2.0
radius_step = 1.2
offset = 0.18                 # décalage intra-muscle
colors = plt.cm.tab10.colors

# =========================
# Tracé
# =========================
for i, muscle in enumerate(muscles):
    r = base_radius + i * radius_step
    color = colors[i]

    # Étude 1 (extérieur)
    start, end = study1[muscle]
    theta = angular_segment(start, end)
    ax.plot(theta, np.full_like(theta, r + offset),
            linewidth=5,
            color=color,
            solid_capstyle="round")

    # Étude 2 (intérieur)
    start, end = study2[muscle]
    theta = angular_segment(start, end)
    ax.plot(theta, np.full_like(theta, r - offset),
            linewidth=3,
            linestyle="--",
            color=color)

# =========================
# Mise en forme
# =========================
ax.set_ylim(0, base_radius + len(muscles) * radius_step)

ax.set_title(
    "Comparaison des activations musculaires en pédalage à bras\n"
    "Cycle de manivelle (0–360°)",
    fontsize=13,
    pad=30
)

# =========================
# LÉGENDES
# =========================

# Légende muscles (couleurs)
muscle_legend = [
    Line2D([0], [0], color=colors[i], lw=4, label=muscle)
    for i, muscle in enumerate(muscles)
]

# Légende études (styles)
study_legend = [
    Line2D([0], [0], color="black", lw=5,
           label="Étude 1 – Ahlers & Jakobsen, (2016)"),
    Line2D([0], [0], color="black", lw=3, linestyle="--",
           label="Étude 2 – Quittman, (2020)")
]

# Ajout des deux légendes
leg1 = ax.legend(handles=muscle_legend,
                 title="Muscles",
                 loc="center left",
                 bbox_to_anchor=(1.05, 0.65))

ax.add_artist(leg1)

ax.legend(handles=study_legend,
          title="Études",
          loc="center left",
          bbox_to_anchor=(1.05, 0.35))

plt.tight_layout()
plt.show()