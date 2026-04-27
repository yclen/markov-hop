import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

LOG_Z = False

csv_path = os.path.join(os.path.dirname(__file__), "migration_times.csv")
df = pd.read_csv(csv_path)

kh_values = np.sort(df["kh"].unique())
nm_values = np.sort(df["n_mediators"].unique()).astype(int)
pivot = df.pivot(index="n_mediators", columns="kh", values="mean_t")
KH, NM = np.meshgrid(kh_values, nm_values)


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


N_data  = df["n_mediators"].values
kh_data = df["kh"].values
t_data  = df["mean_t"].values

# ── Model 1: mean_t = a · N^b · kh^c  (free power law) ───────────────
log_t  = np.log(t_data)
X1 = np.column_stack([np.ones_like(log_t), np.log(N_data), np.log(kh_data)])
cf1, *_ = np.linalg.lstsq(X1, log_t, rcond=None)
a1, b1, c1 = np.exp(cf1[0]), cf1[1], cf1[2]
fit1 = a1 * N_data**b1 * kh_data**c1
r2_1 = r2(t_data, fit1)
fit1_surf = a1 * NM**b1 * KH**c1

print("── Model 1: mean_t = a · N^b · kh^c ──────────────────────")
print(f"  mean_t = {a1:.4f} · N^{b1:.4f} · kh^{c1:.4f}")
print(f"  R² (overall): {r2_1:.6f}")

# ── Model 2: mean_t = a · (N + d)^2 / kh  (offset quadratic) ─────────
# kh^-1 is fixed; fit a and d in: mean_t * kh = a * (N + d)^2
t_times_kh = t_data * kh_data

def offset_quad(N, a, d):
    return a * (N + d)**2

(a2, d2), _ = curve_fit(offset_quad, N_data, t_times_kh, p0=[0.1, 1.0])
fit2 = offset_quad(N_data, a2, d2) / kh_data
r2_2 = r2(t_data, fit2)
fit2_surf = offset_quad(NM, a2, d2) / KH

print("\n── Model 2: mean_t = a · (N + d)² / kh  (offset quadratic) ──")
print(f"  mean_t = {a2:.4f} · (N + {d2:.4f})² / kh")
print(f"  R² (overall): {r2_2:.6f}")

# slices
fixed_kh = kh_values[np.argmin(np.abs(kh_values - 10.0))]
sl_kh = df[df["kh"] == fixed_kh].sort_values("n_mediators")
N_sl  = sl_kh["n_mediators"].values

fit1_kh = a1 * N_sl**b1 * fixed_kh**c1
fit2_kh = offset_quad(N_sl, a2, d2) / fixed_kh
r2_1_kh = r2(sl_kh["mean_t"].values, fit1_kh)
r2_2_kh = r2(sl_kh["mean_t"].values, fit2_kh)

sl_nm  = df[df["n_mediators"] == 5].sort_values("kh")
kh_sl  = sl_nm["kh"].values
fit1_nm = a1 * 5**b1 * kh_sl**c1
fit2_nm = offset_quad(5, a2, d2) / kh_sl
r2_1_nm = r2(sl_nm["mean_t"].values, fit1_nm)
r2_2_nm = r2(sl_nm["mean_t"].values, fit2_nm)

print(f"\n  R² (t vs N,  kh={fixed_kh:.3g}):  M1={r2_1_kh:.6f}  M2={r2_2_kh:.6f}")
print(f"  R² (t vs kh, N=5):          M1={r2_1_nm:.6f}  M2={r2_2_nm:.6f}")

# ── 3D surface ────────────────────────────────────────────────────────
def z(arr):
    return np.log10(arr) if LOG_Z else arr

fig1 = plt.figure(figsize=(11, 7))
ax = fig1.add_subplot(111, projection="3d")
ax.plot_surface(np.log10(KH), NM, z(pivot.values),
                cmap="viridis", linewidth=0, antialiased=True, alpha=0.8)
ax.plot_surface(np.log10(KH), NM, z(fit1_surf),
                color="red", linewidth=0, antialiased=True, alpha=0.2)
ax.plot_surface(np.log10(KH), NM, z(fit2_surf),
                color="cyan", linewidth=0, antialiased=True, alpha=0.2)
ax.set_xlabel("log₁₀(k_h)")
ax.set_ylabel("N mediators")
ax.set_zlabel("log₁₀(mean migration time)" if LOG_Z else "Mean migration time (ns)")
ax.set_title(f"M1 (red) R²={r2_1:.4f}  |  M2 (cyan) R²={r2_2:.4f}")

# ── t vs N at fixed kh ≈ 10 ──────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(N_sl, sl_kh["mean_t"].values,
         marker="o", linewidth=2, color="#4a90d9", label="data")
ax2.plot(N_sl, fit1_kh, linewidth=1.5, linestyle="--", color="red",
         label=f"M1: N^b  R²={r2_1_kh:.4f}")
ax2.plot(N_sl, fit2_kh, linewidth=1.5, linestyle="--", color="cyan",
         label=f"M2: (N+d)²  R²={r2_2_kh:.4f}")
ax2.set_xlabel("N mediators")
ax2.set_ylabel("Mean migration time (ns)")
ax2.set_title(f"Migration time vs N  |  k_h={fixed_kh:.3g} ns⁻¹")
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── t vs kh at fixed N = 5 ───────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.plot(kh_sl, sl_nm["mean_t"].values,
         marker="o", linewidth=2, color="#f5a623", label="data")
ax3.plot(kh_sl, fit1_nm, linewidth=1.5, linestyle="--", color="red",
         label=f"M1: kh^c  R²={r2_1_nm:.4f}")
ax3.plot(kh_sl, fit2_nm, linewidth=1.5, linestyle="--", color="cyan",
         label=f"M2: 1/kh  R²={r2_2_nm:.4f}")
ax3.set_xscale("log")
ax3.set_xlabel("k_h (ns⁻¹)")
ax3.set_ylabel("Mean migration time (ns)")
ax3.set_title("Migration time vs k_h  |  N=5")
ax3.legend()
ax3.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()
