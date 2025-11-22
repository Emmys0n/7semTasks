import numpy as np
import matplotlib.pyplot as plt



# 1. Генерация данных (4 гауссовских класса)
def generate_gaussian_classes(M=4, Ni=50, dm=4.0,
                              romin=-0.9, romax=0.9,
                              random_state=0):
    """
    Генерирует выборку из M двумерных гауссовских классов.

    M      - число классов
    Ni     - число объектов в каждом классе
    dm     - смещение центров между классами
    romin, romax - диапазон для коэффициента корреляции
    """
    rng = np.random.default_rng(random_state)

    means_all = np.array([
        [0.0, 0.0],
        [0.0, dm],
        [dm, 0.0],
        [dm, dm],
        [-dm, -dm],
    ])
    means = means_all[:M]

    X_list = []
    y_list = []

    for i in range(M):
        ro = rng.uniform(romin, romax)
        cov = np.array([
            [1.0, ro],
            [ro, 1.0]
        ])

        Xi = rng.multivariate_normal(mean=means[i],
                                     cov=cov,
                                     size=Ni)
        yi = np.full(Ni, i, dtype=int)

        X_list.append(Xi)
        y_list.append(yi)

    X = np.vstack(X_list)       # (N, 2)
    y = np.concatenate(y_list)  # (N,)

    return X, y



# 2. Функция расстояний
def compute_distances(X, centers, metric='sqeuclidean', eps=1e-12):
    """
    Считает матрицу расстояний между объектами X и центрами centers
    для одной из метрик:
      - 'sqeuclidean' : сумма квадратов разностей
      - 'cityblock'   : манхэттенское расстояние (L1)
      - 'cosine'      : косинусная мера (1 - cos similarity)
      - 'correlation' : корреляционная мера (1 - corr)
    """
    X = np.asarray(X)
    centers = np.asarray(centers)

    N, d = X.shape
    k = centers.shape[0]

    D = np.zeros((N, k))

    if metric == 'sqeuclidean':
        for j in range(k):
            diff = X - centers[j]
            D[:, j] = np.sum(diff ** 2, axis=1)

    elif metric == 'cityblock':
        for j in range(k):
            diff = np.abs(X - centers[j])
            D[:, j] = np.sum(diff, axis=1)

    elif metric == 'cosine':
        x_norm = np.linalg.norm(X, axis=1) + eps
        c_norm = np.linalg.norm(centers, axis=1) + eps
        for j in range(k):
            sim = (X @ centers[j]) / (x_norm * c_norm[j])
            D[:, j] = 1.0 - sim

    elif metric == 'correlation':
        # 1 - corr(x, c)
        Xc = X - X.mean(axis=1, keepdims=True)
        x_norm = np.sqrt((Xc ** 2).sum(axis=1)) + eps

        centers_c = centers - centers.mean(axis=1, keepdims=True)
        c_norm = np.sqrt((centers_c ** 2).sum(axis=1)) + eps

        for j in range(k):
            num = Xc @ centers_c[j]
            den = x_norm * c_norm[j] + eps
            corr = num / den
            D[:, j] = 1.0 - corr

    else:
        raise ValueError(f"Неизвестная метрика: {metric}")

    return D



# 3. K-means с выбором метрики
def kmeans_custom(X, n_clusters, metric='sqeuclidean',
                  n_init=5, max_iter=100, tol=1e-4,
                  random_state=None):
    """
    Простая реализация k-means с поддержкой разных метрик расстояния.
    Возвращает:
      labels  - метки кластеров для каждого объекта
      centers - найденные центры кластеров
      inertia - суммарное расстояние внутри кластеров
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    N = X.shape[0]

    best_inertia = None
    best_labels = None
    best_centers = None

    for init in range(n_init):
        # Случайная инициализация центров из объектов
        init_idx = rng.choice(N, size=n_clusters, replace=False)
        centers = X[init_idx].copy()

        for it in range(max_iter):
            # 1) присвоение точек ближайшему центру
            D = compute_distances(X, centers, metric=metric)
            labels = np.argmin(D, axis=1)

            # 2) пересчёт центров как средних по кластерам
            new_centers = np.zeros_like(centers)
            for k in range(n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centers[k] = cluster_points.mean(axis=0)
                else:
                    # пустой кластер -> случайная точка
                    new_centers[k] = X[rng.integers(0, N)]

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers

            if shift < tol:
                break

        D_final = compute_distances(X, centers, metric=metric)
        inertia = np.min(D_final, axis=1).sum()

        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia



# 4. Оценка качества кластеризации
def clustering_quality(y_true, y_pred, n_classes):
    """
    Строит "путаницу" класс-кластер и подбирает
    соответствие кластеров реальным классам (жадно по максимуму).

    Возвращает:
      mapping_cluster_to_class - dict: кластер -> класс
      prM                      - индекс качества (доля верных)
      ercl                     - частота ошибок (1 - prM)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    N = len(y_true)
    K = int(y_pred.max()) + 1   # число кластеров по максимуму метки

    conf = np.zeros((n_classes, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1

    # Жадное соответствие класс <-> кластер
    pairs = []
    for c in range(n_classes):
        for k in range(K):
            pairs.append((conf[c, k], c, k))
    pairs.sort(reverse=True, key=lambda x: x[0])

    mapping_class_to_cluster = {}
    mapping_cluster_to_class = {}

    for cnt, c, k in pairs:
        if cnt == 0:
            continue
        if c not in mapping_class_to_cluster and k not in mapping_cluster_to_class:
            mapping_class_to_cluster[c] = k
            mapping_cluster_to_class[k] = c
        if len(mapping_cluster_to_class) == n_classes:
            break

    # Переводим кластеры в "предсказанные классы"
    y_aligned = np.empty_like(y_pred)
    for i, cl in enumerate(y_pred):
        if cl in mapping_cluster_to_class:
            y_aligned[i] = mapping_cluster_to_class[cl]
        else:
            y_aligned[i] = 0

    correct = np.sum(y_true == y_aligned)
    prM = correct / N
    ercl = 1.0 - prM

    return mapping_cluster_to_class, prM, ercl

# 5. Основной запуск
if __name__ == "__main__":
    M = 4          # число классов / кластеров
    Ni = 50        # количество точек в каждом классе
    dm = 4.0

    X, y_true = generate_gaussian_classes(M=M, Ni=Ni, dm=dm,
                                          romin=-0.9, romax=0.9,
                                          random_state=0)

    print("X shape:", X.shape)

    metrics = ['sqeuclidean', 'cityblock', 'cosine', 'correlation']
    results = []

    # считаем k-means для каждой метрики
    for metric in metrics:
        labels, centers, inertia = kmeans_custom(
            X, n_clusters=M,
            metric=metric,
            n_init=10,
            max_iter=100,
            tol=1e-4,
            random_state=42
        )

        mapping, prM, ercl = clustering_quality(y_true, labels, n_classes=M)

        print(f"\nМетрика расстояния: {metric}")
        print(f"  Индекс качества кластеризации prM = {prM:.4f}")
        print(f"  Частота ошибок ercl = {ercl:.4f}")
        if metric == 'correlation':
            print(f"  Inertia (суммарное внутрикластерное расстояние) = {inertia:.4e}")
        else:
            print(f"  Inertia (суммарное внутрикластерное расстояние) = {inertia:.4f}")

        results.append((metric, prM, ercl, labels, centers, mapping))

    
    # 6. Визуализация в виде схемы
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0])

    # ---- верхний центральный график: истинные классы ----
    ax_true = fig.add_subplot(gs[0, :])   # занимает всю строку
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for cls in range(M):
        ax_true.scatter(X[y_true == cls, 0],
                        X[y_true == cls, 1],
                        s=40,
                        alpha=0.8,
                        label=f"Класс {cls + 1}")
    ax_true.set_title("Исходные данные (истинные классы)")
    ax_true.set_xlabel("x1")
    ax_true.set_ylabel("x2")
    ax_true.grid(True)
    ax_true.legend(loc='best')

    # вспомогательная функция рисования одного из нижних графиков
    def plot_metric_ax(metric_name, ax):
        metric, prM, ercl, labels, centers, mapping = next(
            r for r in results if r[0] == metric_name
        )

        # переводим кластеры в классы для поиска ошибок
        y_aligned = np.array([mapping.get(cl, 0) for cl in labels])
        mis = (y_true != y_aligned)

        # точки по кластерам
        for k in range(M):
            ax.scatter(X[labels == k, 0],
                       X[labels == k, 1],
                       s=40,
                       alpha=0.8,
                       label=f"Кластер {k + 1}")
        # центры
        ax.scatter(centers[:, 0],
                   centers[:, 1],
                   marker='*',
                   s=150,
                   c='k',
                   label="Центры")

        # ошибки (поверх) – пустые кружки
        ax.scatter(X[mis, 0],
                   X[mis, 1],
                   s=80,
                   facecolors='none',
                   edgecolors='k',
                   linewidths=1.5,
                   label="Ошибки")

        ax.set_title(f"k-means ({metric_name})\nprM={prM:.3f}, ercl={ercl:.3f}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True)
        ax.legend(loc='best')

    # нижние 4 графика: как на твоем рисунке
    ax_sq = fig.add_subplot(gs[1, 0])  # слева вторая строка
    ax_ct = fig.add_subplot(gs[1, 1])  # справа вторая строка
    ax_co = fig.add_subplot(gs[2, 0])  # слева третья строка
    ax_cr = fig.add_subplot(gs[2, 1])  # справа третья строка

    plot_metric_ax('sqeuclidean', ax_sq)
    plot_metric_ax('cityblock',   ax_ct)
    plot_metric_ax('cosine',      ax_co)
    plot_metric_ax('correlation', ax_cr)

    plt.tight_layout()
    plt.show()
