import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Hill-Thelen (biorbd) params
# ----------------------------
kpe = 5.0
e0  = 0.6

kvce = 0.06
flen = 1.6
a = 3/11  # Thelen concentric params in biorbd code
b = 3/11

# ----------------------------
# Vectorized muscle functions
# ----------------------------
def flce(l):
    """Active force–length (HillThelenType.cpp): exp(-((l-1)^2)/0.45)"""
    l = np.asarray(l, dtype=float)
    return np.exp(-((l - 1.0) ** 2) / 0.45)

def flpe(l):
    """
    Passive force–length (HillThelenType.cpp):
      0 if l<=1
      (exp(kpe*(l-1)/e0)-1)/(exp(kpe)-1) if l>1
    """
    l = np.asarray(l, dtype=float)
    out = np.zeros_like(l)
    mask = l > 1.0
    out[mask] = (np.exp(kpe * (l[mask] - 1.0) / e0) - 1.0) / (np.exp(kpe) - 1.0)
    return out

def fvce(v):
    """
    Active force–velocity (HillThelenType.cpp):
      if v >= 0: (1 + v*flen/kvce)/(1 + v/kvce)
      elif v >= -1: (1+a)*b/(-v + b) - a
      else: 0
    """
    v=v*(10*0.1)
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v)

    # eccentric / lengthening
    mask_ecc = v >= 0.0
    out[mask_ecc] = (1.0 + v[mask_ecc] * flen / kvce) / (1.0 + v[mask_ecc] / kvce)

    # concentric / shortening (limited to [-1,0)
    mask_con = (v >= -1.0) & (v < 0.0)
    out[mask_con] = (1.0 + a) * b / (-v[mask_con] + b) - a

    # v < -1 stays 0
    return out

# ----------------------------
# "Interesting" intervals
# ----------------------------
l_grid = np.linspace(0.5, 1.8, 600)     # around optimal length + passive region
v_grid = np.linspace(-1.3, 2.0, 800)    # show cutoff (<-1), concentric, eccentric saturation

# ----------------------------
# Plot: 3 subplots
# ----------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

# FlCE
ax[0].plot(l_grid, flce(l_grid), linewidth=2)
ax[0].axvline(1.0, linestyle="--", linewidth=1)
ax[0].set_title("FlCE(l̃) actif")
ax[0].set_xlabel("l̃ = Lm/Lopt")
ax[0].set_ylabel("FlCE [-]")
ax[0].grid(True, alpha=0.3)

# FlPE
ax[1].plot(l_grid, flpe(l_grid), linewidth=2)
ax[1].axvline(1.0, linestyle="--", linewidth=1)
ax[1].set_title("FlPE(l̃) passif")
ax[1].set_xlabel("l̃ = Lm/Lopt")
ax[1].set_ylabel("FlPE [-]")
ax[1].grid(True, alpha=0.3)

# FvCE
ax[2].plot(v_grid, fvce(v_grid), linewidth=2)
ax[2].axvline(-1.0, linestyle="--", linewidth=1)
ax[2].axvline(0.0, linestyle="--", linewidth=1)
ax[2].set_title("FvCE(ṽ) actif")
ax[2].set_xlabel("ṽ = v / (Lopt * vmax)")
ax[2].set_ylabel("FvCE [-]")
ax[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

