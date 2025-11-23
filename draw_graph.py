import numpy as np
import matplotlib.pyplot as plt

# Loss 불러오기
d = np.load("./logs_hexapod_hardware/student_bc/bc_loss_history.npz")
train = d["train"]
val = d["val"]
epochs = np.arange(1, len(train) + 1)

plt.figure(figsize=(7, 5))

# 🔥 선을 더 두껍고 진하게
plt.plot(epochs, train, label="Train MSE",
         linewidth=3.0, color="#1f77b4")
plt.plot(epochs, val, label="Val MSE",
         linewidth=3.0, color="#d62728")

# ✨ 격자를 연하게 (alpha 낮춤)
plt.grid(True, alpha=0.5, linewidth=0.6)

plt.xlabel("Epoch", fontsize=14)
plt.ylabel("MSE Loss", fontsize=14)
plt.title("BC Training and Validation Loss", fontsize=18)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# ⭐ 범례 글씨 크게 (기존 12 → 15로 증가)
plt.legend(fontsize=15)

plt.tight_layout()
plt.show()
