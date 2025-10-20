# ------------------------------
# Momo vs Gemma Comparative Performance Chart
# ------------------------------
import matplotlib.pyplot as plt
import numpy as np

# ---- Criteria ----
criteria = ["Speed", "Model Size", "Computational Power"]

# ---- Normalized Scores (1–5 scale; 5 = best)
# Assume Gemma is faster, smaller, and uses less compute
momo  = [3.5, 2.8, 3.0]
gemma = [4.8, 4.5, 4.7]

x = np.arange(len(criteria))
width = 0.35

# ---- Create the figure ----
plt.figure(figsize=(8,5))
bars1 = plt.barh(x - width/2, momo, width, color="#003366", label="Momo")
bars2 = plt.barh(x + width/2, gemma, width, color="#B22234", label="Gemma")

# ---- Axes, grid, and labels ----
plt.xlabel("Normalized Performance (1–5)", fontsize=12, weight='bold')
plt.yticks(x, criteria, fontsize=12, weight='bold')
plt.xlim(0,5)
plt.grid(axis='x', linestyle='--', alpha=0.4)

# ---- Annotate bars ----
for bars in [bars1, bars2]:
    for bar in bars:
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.1f}", va='center', fontsize=10)

# ---- Title and legend ----
plt.title("Preliminary Comparison: Momo vs Gemma\n(Normalized 1–5)",
          fontsize=15, weight='bold', pad=10)

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=False,
    fontsize=11
)

plt.tight_layout()
plt.savefig("Momo_vs_Gemma_Comparison.png", dpi=600, bbox_inches="tight")
plt.show()
