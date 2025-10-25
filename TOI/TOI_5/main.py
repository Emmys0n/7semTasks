import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal

#  1) Смесь 5 Гауссов в R^2
rng = np.random.default_rng(42)
N = 5000
weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])

means = np.array([
    [-2.5,  0.0],
    [ 2.0,  2.5],
    [ 2.0, -2.5],
    [ 0.0,  0.0],
    [-4.0,  3.0],
])

covs = np.array([
    [[0.7,  0.2], [0.2, 0.9]],
    [[0.8, -0.3], [-0.3, 0.8]],
    [[0.6,  0.0], [0.0, 0.9]],
    [[1.0,  0.4], [0.4, 0.7]],
    [[0.5, -0.2], [-0.2, 0.6]],
])

comp_idx = rng.choice(len(weights), size=N, p=weights)
X = np.vstack([
    rng.multivariate_normal(means[j], covs[j], size=(comp_idx == j).sum())
    for j in range(len(weights))
])

#  2) Истинная плотность на сетке
# Можно уменьшить сетку, если ОЗУ совсем мало: напр. 140x140
x1 = np.linspace(-6, 6, 160)
x2 = np.linspace(-6, 6, 160)
X1, X2 = np.meshgrid(x1, x2)
grid = np.column_stack([X1.ravel(), X2.ravel()])

p_true = np.zeros(grid.shape[0])
for w, m, C in zip(weights, means, covs):
    p_true += w * multivariate_normal.pdf(grid, mean=m, cov=C)

#  3) kNN: поиск k* по MSE — ПЕРВЫЙ ПРОХОД (только MSE)
ks = np.arange(1, 61)   # диапазон k
max_k = ks[-1]
nbrs = NearestNeighbors(n_neighbors=max_k, algorithm="auto").fit(X)

# аккумулируем сумму квадратов ошибки по батчам (без хранения p_hat)
mse_sums = np.zeros(len(ks), dtype=float)
M = grid.shape[0]
batch = 4000  # уменьшай, если ОЗУ мало (например, 2000)

for i in range(0, M, batch):
    G = grid[i:i+batch]
    p_true_b = p_true[i:i+batch]
    dists_all, _ = nbrs.kneighbors(G, n_neighbors=max_k)  # (batch, max_k)

    # для каждого k берём соответствующий столбец дистанций
    for idx_k, k in enumerate(ks):
        r_k = dists_all[:, k-1]
        V = np.pi * np.maximum(r_k, 1e-12)**2
        p_hat_b = k / (N * V)
        mse_sums[idx_k] += np.sum((p_hat_b - p_true_b)**2)

mse = mse_sums / M
imin = int(np.argmin(mse))
best_k, best_mse = int(ks[imin]), float(mse[imin])
print(f"[ИТОГО] Оптимальный k* = {best_k}, MSE = {best_mse:.6e}")

#  4) kNN-карта для k* — ВТОРОЙ ПРОХОД (только для лучшего k)
best_phat_flat = np.empty(M, dtype=float)
for i in range(0, M, batch):
    G = grid[i:i+batch]
    dists_all, _ = nbrs.kneighbors(G, n_neighbors=best_k)
    r_k = dists_all[:, -1]
    V = np.pi * np.maximum(r_k, 1e-12)**2
    best_phat_flat[i:i+batch] = best_k / (N * V)

p_true = p_true.reshape(X1.shape)
best_phat = best_phat_flat.reshape(X1.shape)

#  5) Визуализация
# График MSE(k) с отметкой минимума
plt.figure()
plt.plot(ks, mse, marker='.')
plt.xlabel("k"); plt.ylabel("MSE"); plt.grid(True)
plt.title("Зависимость MSE kNN-оценки плотности от k (2D, 5 гауссов)")
plt.scatter([best_k], [best_mse], s=80, zorder=3)
plt.annotate(f"k*={best_k}, MSE={best_mse:.2e}",
             (best_k, best_mse), xytext=(best_k+2, best_mse*1.2 + 1e-6),
             arrowprops=dict(arrowstyle="->"))
# при желании расширь масштаб по Y:
# plt.ylim(0, np.percentile(mse, 95)*1.1)
plt.show()

# Контуры: истина vs оценка при k*
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
cs1 = plt.contourf(X1, X2, p_true, levels=15)
plt.colorbar(cs1); plt.title("Истинная плотность")
plt.xlabel("x1"); plt.ylabel("x2")

plt.subplot(1,2,2)
cs2 = plt.contourf(X1, X2, best_phat, levels=15)
plt.colorbar(cs2); plt.title(f"kNN-плотность, k*={best_k}")
plt.xlabel("x1"); plt.ylabel("x2")

plt.tight_layout()
plt.show()
