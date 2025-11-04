import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ФЛАГИ ВИЗУАЛИЗАЦИИ

PLOT_SURFACES = True       # 3D поверхности плотностей (теория vs Парзен)
PLOT_DECISION  = True      # карта решений Парзена
PLOT_SWEEP     = True      # график P(err) vs h
RANDOM_SEED    = 42

# ПАРАМЕТРЫ КЛАССОВ (по ЛР-3)

m1 = np.array([ 2.0, 1.0])
m2 = np.array([-1.0, 1.0])

C1 = np.array([[ 3.0, -1.0],
               [-1.0,  3.0]])

C2 = np.array([[ 5.0,  2.0],
               [ 2.0,  6.0]])

p1 = 0.5
p2 = 0.5

rv1 = multivariate_normal(mean=m1, cov=C1)
rv2 = multivariate_normal(mean=m2, cov=C2)

rng = np.random.default_rng(RANDOM_SEED)

# ОБЪЁМЫ ВЫБОРОК
N_train_per_class = 1000
N_test_per_class  = 5000

# ВСПОМОГАТЕЛЬНОЕ

def add_cov_ellipse(ax, mean, cov, color, alpha=0.25):
    """Эллипс 95% (χ²_2(0.95)≈5.991) для наглядности."""
    vals, vecs = np.linalg.eigh(cov)
    width  = 2*np.sqrt(5.991*vals[1])
    height = 2*np.sqrt(5.991*vals[0])
    angle = np.degrees(np.arctan2(vecs[1,1], vecs[0,1]))
    ax.add_patch(Ellipse(mean, width, height, angle=angle,
                         facecolor=color, edgecolor=color, alpha=alpha))

def scott_bandwidth_isotropic(X):
    """
    h = sigma * n^(-1/(d+4)), где sigma — среднее СКО по осям.
    (правило Скотта, изотропное)
    """
    n, d = X.shape
    sigma = X.std(axis=0, ddof=1).mean()
    h = sigma * n ** (-1.0/(d+4.0))
    return float(h)

def parzen_pdf_generic(X_train, h, kernel="gaussian"):
    """
    Универсальная Парзен-оценка p_hat(x) в 2D с изотропным h.
    kernel: 'gaussian' | 'epan' | 'uniform'
    """
    X = np.asarray(X_train, float)
    n, d = X.shape
    norm = 1.0 / (n * (h ** d))

    def K_gauss(r2):  # r2 = ||(x-xi)/h||^2
        return (1.0 / ((2*np.pi)**(d/2))) * np.exp(-0.5 * r2)

    def K_epan(r2):
        # Epanechnikov: K(u) = c_d * (1 - ||u||^2) for ||u||<=1 else 0
        # c_d для d=2: 2/π
        r = np.sqrt(r2)
        val = np.maximum(0.0, 1.0 - r**2)
        return (2/np.pi) * val

    def K_uniform(r2):
        # Uniform (прямоугольное): K(u) = c_d for ||u||<=1 else 0
        # c_d для d=2: 1/π
        r = np.sqrt(r2)
        return (1/np.pi) * (r <= 1.0)

    if kernel == "gaussian":
        K = K_gauss
    elif kernel in ("epan", "epanechnikov"):
        K = K_epan
    elif kernel in ("uniform", "rect"):
        K = K_uniform
    else:
        raise ValueError("kernel must be 'gaussian' | 'epan' | 'uniform'")

    def pdf(P):
        P = np.atleast_2d(P)
        P2 = np.sum(P**2, axis=1, keepdims=True)
        X2 = np.sum(X**2, axis=1, keepdims=True).T
        d2 = (P2 + X2 - 2.0*(P @ X.T)) / (h**2)
        return norm * K(d2).sum(axis=1)

    return pdf

def classify_parzen(X, pdf1, pdf2, p1, p2):
    """Байесовское правило решения поверх оценённых плотностей."""
    P1 = pdf1(X) * p1
    P2 = pdf2(X) * p2
    return np.where(P1 >= P2, 1, 2)

def classify_bayes_true(X, rv1, rv2, p1, p2):
    """Байес (истинные плотности из ЛР-3)."""
    P1 = rv1.pdf(X) * p1
    P2 = rv2.pdf(X) * p2
    return np.where(P1 >= P2, 1, 2)

def compute_metrics(cm):
    """Из матрицы ошибок возвращает (acc, perr, alpha1, beta2)."""
    acc = np.trace(cm) / cm.sum()
    perr = 1.0 - acc
    alpha1 = cm[0,1] / cm[0].sum()  # P(реш.2 | ист.1)
    beta2  = cm[1,0] / cm[1].sum()  # P(реш.1 | ист.2)
    return acc, perr, alpha1, beta2



# ДАННЫЕ: train/test

X1_train = rv1.rvs(size=N_train_per_class, random_state=rng)
X2_train = rv2.rvs(size=N_train_per_class, random_state=rng)

X1_test = rv1.rvs(size=N_test_per_class,  random_state=rng)
X2_test = rv2.rvs(size=N_test_per_class,  random_state=rng)
X_test  = np.vstack([X1_test, X2_test])
y_true  = np.array([1]*N_test_per_class + [2]*N_test_per_class)


# БАЗОВОЕ СРАВНЕНИЕ: Байес (истина) vs Парзен (гаусс + Scott)

# Байес (истинные)
y_bayes = classify_bayes_true(X_test, rv1, rv2, p1, p2)
cm_bayes = confusion_matrix(y_true, y_bayes, labels=[1,2])
acc_bayes, perr_bayes, a1_bayes, b2_bayes = compute_metrics(cm_bayes)

# Парзен: h по Скотту для каждого класса
h1 = scott_bandwidth_isotropic(X1_train)
h2 = scott_bandwidth_isotropic(X2_train)
pdf1_hat = parzen_pdf_generic(X1_train, h1, kernel="gaussian")
pdf2_hat = parzen_pdf_generic(X2_train, h2, kernel="gaussian")
y_parzen = classify_parzen(X_test, pdf1_hat, pdf2_hat, p1, p2)
cm_parzen = confusion_matrix(y_true, y_parzen, labels=[1,2])
acc_parzen, perr_parzen, a1_parzen, b2_parzen = compute_metrics(cm_parzen)

print(" СРАВНЕНИЕ КЛАССИФИКАТОРОВ ")
print(f"[Байес (истинные)]    acc={acc_bayes:.4f}  P(err)={perr_bayes:.4f}  "
      f"alpha1={a1_bayes:.4f}  beta2={b2_bayes:.4f}")
print(f"[Парзен (Gauss+Scott)] acc={acc_parzen:.4f}  P(err)={perr_parzen:.4f}  "
      f"alpha1={a1_parzen:.4f}  beta2={b2_parzen:.4f}")
print("\nМатрицы ошибок [истина по строкам 1/2, решение по столбцам 1/2]:")
print("Байес:\n", cm_bayes)
print("Парзен (Gauss+Scott):\n", cm_parzen)

# СЕТКА ДЛЯ ПОВЕРХНОСТЕЙ/КАРТЫ РЕШЕНИЙ

all_data = np.vstack([X1_train, X2_train, X_test])
x_min, x_max = all_data[:,0].min()-2.0, all_data[:,0].max()+2.0
y_min, y_max = all_data[:,1].min()-2.0, all_data[:,1].max()+2.0
xs = np.linspace(x_min, x_max, 120)
ys = np.linspace(y_min, y_max, 120)
XX, YY = np.meshgrid(xs, ys)
GRID = np.stack([XX.ravel(), YY.ravel()], axis=1)

# теоретические и Парзен-плотности для поверхностей
f1_true = rv1.pdf(GRID).reshape(XX.shape)
f2_true = rv2.pdf(GRID).reshape(XX.shape)
f1_hat  = pdf1_hat(GRID).reshape(XX.shape)
f2_hat  = pdf2_hat(GRID).reshape(XX.shape)


# ВИЗУАЛИЗАЦИЯ: ПОВЕРХНОСТИ

if PLOT_SURFACES:
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(XX, YY, f1_true, rstride=2, cstride=2, alpha=0.9, cmap=cm.viridis)
    ax.set_title("Класс 1 — теоретическая плотность")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("p(x|w1)")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(XX, YY, f1_hat, rstride=2, cstride=2, alpha=0.9, cmap=cm.viridis)
    ax.set_title(f"Класс 1 — Парзен (Gauss, h1={h1:.3f})")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel(r"$\hat p(x|w1)$")
    plt.tight_layout(); plt.show()

    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(XX, YY, f2_true, rstride=2, cstride=2, alpha=0.9, cmap=cm.plasma)
    ax.set_title("Класс 2 — теоретическая плотность")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("p(x|w2)")

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(XX, YY, f2_hat, rstride=2, cstride=2, alpha=0.9, cmap=cm.plasma)
    ax.set_title(f"Класс 2 — Парзен (Gauss, h2={h2:.3f})")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel(r"$\hat p(x|w2)$")
    plt.tight_layout(); plt.show()


# ВИЗУАЛИЗАЦИЯ: КАРТА РЕШЕНИЙ ПАРЗЕНА

if PLOT_DECISION:
    P1_grid = f1_hat * p1
    P2_grid = f2_hat * p2
    Z = np.where(P1_grid >= P2_grid, 1, 2)

    plt.figure(figsize=(7,6))
    plt.contourf(XX, YY, Z, levels=[0.5,1.5,2.5], colors=["#ffaaaa","#aaaaff"], alpha=0.35)
    plt.contour(XX, YY, Z, levels=[1.5], colors='k', linestyles='--', linewidths=2, alpha=0.9)
    plt.scatter(X1_train[:,0], X1_train[:,1], s=6, c='red',  alpha=0.55, label='train w1')
    plt.scatter(X2_train[:,0], X2_train[:,1], s=6, c='blue', alpha=0.55, label='train w2')
    add_cov_ellipse(plt.gca(), m1, C1, 'red',  0.18)
    add_cov_ellipse(plt.gca(), m2, C2, 'blue', 0.18)
    plt.title("Карта решений Парзена (гаусс, изотропное окно)")
    plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

# ВИЗУАЛИЗАЦИЯ МАТРИЦЫ ОШИБОК
fig, axes = plt.subplots(1, 2, figsize=(9,4))
for ax, cmx, ttl in zip(axes,
                        [cm_bayes, cm_parzen],
                        [f"QDA (истинные)\nacc={acc_bayes:.3f}, P(err)={perr_bayes:.3f}",
                         f"Parzen (Gauss+Scott)\nacc={acc_parzen:.3f}, P(err)={perr_parzen:.3f}"]):
    sns.heatmap(cmx, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Предсказанный"); ax.set_ylabel("Истинный"); ax.set_title(ttl)
plt.tight_layout(); plt.show()

# КВ-1: СКАНИРОВАНИЕ h -> P(err)
# (общий h для обоих классов, ядро гауссовское)

# Возьмём средний из (h1, h2) и сканируем вокруг него
h_scott_mean = 0.5 * (h1 + h2)
hs = np.linspace(max(0.15, h_scott_mean*0.5), h_scott_mean*1.8, 12)

results = []
for h in hs:
    pdf1 = parzen_pdf_generic(X1_train, h, kernel="gaussian")
    pdf2 = parzen_pdf_generic(X2_train, h, kernel="gaussian")
    y_hat = classify_parzen(X_test, pdf1, pdf2, p1, p2)
    cm = confusion_matrix(y_true, y_hat, labels=[1,2])
    acc, perr, a1, b2 = compute_metrics(cm)
    results.append((h, perr, acc, a1, b2))

print("\n СКАНИРОВАНИЕ h (ядро: gaussian, общий h для классов) ")
best = None
for h, perr, acc, a1, b2 in results:
    print(f"h={h:.3f}  P(err)={perr:.4f}  acc={acc:.4f}  alpha1={a1:.4f}  beta2={b2:.4f}")
    if best is None or perr < best[1]:
        best = (h, perr, acc, a1, b2)
print(f"Лучшее: h={best[0]:.3f}  P(err)={best[1]:.4f}  acc={best[2]:.4f}  "
      f"alpha1={best[3]:.4f}  beta2={best[4]:.4f}")

if PLOT_SWEEP:
    plt.figure(figsize=(7,4))
    plt.plot([r[0] for r in results], [r[1] for r in results], marker='o')
    plt.xlabel("h"); plt.ylabel("P(err)")
    plt.title("Сканирование h (ядро Gauss): P(err) vs h")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

# КВ-2: СРАВНЕНИЕ ЯДЕР (подбираем лучший h для каждого ядра)

kernels = ["gaussian", "epan", "uniform"]
print("\n СРАВНЕНИЕ ЯДЕР (best h и P(err)) ")
for ker in kernels:
    best_k = None
    for h in hs:
        pdf1 = parzen_pdf_generic(X1_train, h, kernel=ker)
        pdf2 = parzen_pdf_generic(X2_train, h, kernel=ker)
        y_hat = classify_parzen(X_test, pdf1, pdf2, p1, p2)
        cm = confusion_matrix(y_true, y_hat, labels=[1,2])
        acc, perr, a1, b2 = compute_metrics(cm)
        if best_k is None or perr < best_k[1]:
            best_k = (h, perr, acc, a1, b2)
    print(f"{ker:8s}  best h={best_k[0]:.3f}  P(err)={best_k[1]:.4f}  "
          f"acc={best_k[2]:.4f}  alpha1={best_k[3]:.4f}  beta2={best_k[4]:.4f}")

