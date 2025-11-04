import numpy as np
import matplotlib.pyplot as plt

# 参数
A1 = 0.815
phi1 = 3.462
A2 = 0.860
phi2 = 5.854
omega = 2 * np.pi

# 时间轴 (0~4s)
t = np.linspace(0, 4, 2000)

# 速度函数
v = -(A1 * np.sin(omega * t + phi1 + np.pi) + A2 * np.sin(2 * omega * t + phi2 + np.pi))

# 绘图
plt.figure(figsize=(8, 3))
plt.plot(t, v, color='darkgreen', linewidth=2)
plt.title(r"$v(t) = -[A_1 \sin(\omega t + \phi_1 + \pi) + A_2 \sin(2\omega t + \phi_2 + \pi)]$", fontsize=11)
plt.xlabel("Time (s)")
plt.ylabel("v(t)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
