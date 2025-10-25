import numpy as np
import matplotlib.pyplot as plt

# ПАРАМЕТРЫ
pI = 0.6                                  # вероятность искажения бита
rng = np.random.default_rng(42)           # генератор для воспроизводимости

# Эксперимент: сколько искажений на класс
N_TRIALS_PER_CLASS = 6000

THEOR_N = 200_000
THEOR_BATCH = 10_000

#БУКВЫ P / M / I
P_8x8 = np.array([
    [1,1,1,1,0,0,0,0],
    [1,0,0,0,1,0,0,0],
    [1,1,1,1,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
], dtype=int)

M_8x8 = np.array([
    [1,0,0,0,0,0,0,1],
    [1,1,0,0,0,0,1,1],
    [1,0,1,0,0,1,0,1],
    [1,0,0,1,1,0,0,1],
    [1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1],
], dtype=int)

I_8x8 = np.array([
    [1,1,1,1,1,1,1,1],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [1,1,1,1,1,1,1,1],
], dtype=int)

templates = np.stack([P_8x8.flatten(), M_8x8.flatten(), I_8x8.flatten()], axis=0)
class_names = ['P', 'M', 'I']
K, n = templates.shape  # K=3, n=64

# ВИЗУАЛИЗАЦИЯ БУКВ (Ч/Б)
def plot_letters_bw(letter_mats, save=False, save_prefix="letter_"):
    fig, axes = plt.subplots(1, len(letter_mats), figsize=(9, 3))

    if len(letter_mats) == 1:
        axes = [axes]

    for ax, (name, mat) in zip(axes, letter_mats):
        # Рисуем пиксели строго в диапазоне [-0.5, 7.5] × [-0.5, 7.5]
        im = ax.imshow(
            mat, cmap='gray_r', vmin=0, vmax=1,
            interpolation='nearest', origin='upper',
            extent=(-0.5, 7.5, 7.5, -0.5)
        )

        ax.set_xticks(range(8))
        ax.set_yticks(range(8))

        # Границы клеток — это minor-типы по целым границам (-0.5..7.5 с шагом 1)
        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)

        # Сетка только по границам клеток (minor grid)
        ax.grid(which='minor', color='lightgray', linewidth=0.8)
        ax.grid(which='major', visible=False)

        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Колонка")
        ax.set_ylabel("Строка")
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(7.5, -0.5)  # чтобы нулевая строка была сверху
        ax.set_aspect('equal')

        if save:
            plt.imsave(f"{save_prefix}{name}.png", mat, cmap='gray_r', vmin=0, vmax=1)

    fig.suptitle("Бинарные шаблоны букв (0/1 → черно-белое)", fontsize=16)
    plt.tight_layout()
    plt.show()

# Функция искажения (бит переворачивается с вероятностью p)
def distort_pattern(pattern, p):
    noise = (rng.random(pattern.shape) < p).astype(int)
    return np.bitwise_xor(pattern, noise)

# Искажаем все три
P_d = distort_pattern(P_8x8, pI)
M_d = distort_pattern(M_8x8, pI)
I_d = distort_pattern(I_8x8, pI)

# ФОРМУЛА (5.36)
# g''(x) = (L10+P01)*ln((1-p)/p) + (n - L10 - P01)*ln(p/(1-p)) + ln P(ω)
# где L10 = #(x=1, t=0), P01 = #(x=0, t=1)
def counts_L10_P01(x: np.ndarray, t: np.ndarray):
    L10 = np.sum((x == 1) & (t == 0))
    P01 = np.sum((x == 0) & (t == 1))
    return int(L10), int(P01)

def g536(x: np.ndarray, t: np.ndarray, p_flip: float, prior: float) -> float:
    L10, P01 = counts_L10_P01(x, t)
    if p_flip == 0.0:
        base = 0.0 if (L10 + P01) == 0 else -np.inf
    elif p_flip == 1.0:
        base = 0.0 if (L10 + P01) == n else -np.inf
    else:
        d = L10 + P01
        base = d * np.log((1 - p_flip) / p_flip) + (n - d) * np.log(p_flip / (1 - p_flip))
    return base + np.log(prior)

def classify_536(x: np.ndarray, templates: np.ndarray, priors: np.ndarray, p_flip: float) -> int:
    scores = [g536(x, templates[i], p_flip, priors[i]) for i in range(templates.shape[0])]
    return int(np.argmax(scores))

# ВСПОМОГАТЕЛЬНЫЕ
def corrupt(template: np.ndarray, p_flip: float, rng: np.random.Generator) -> np.ndarray:
    flips = rng.random(template.shape[0]) < p_flip
    return np.bitwise_xor(template, flips.astype(int))



# МАТРИЦЫ ОШИБОК
def experimental_confusion_matrix(templates, priors, p_flip, N, rng):
    K = templates.shape[0]
    counts = np.zeros((K, K), dtype=int)
    for true_idx in range(K):
        t_true = templates[true_idx]
        for _ in range(N):
            x = corrupt(t_true, p_flip, rng)
            pred = classify_536(x, templates, priors, p_flip)
            counts[true_idx, pred] += 1
    return counts / counts.sum(axis=1, keepdims=True)

def theoretical_confusion_matrix(templates, priors, p_flip, N=THEOR_N, batch=THEOR_BATCH, seed=12345):

    K, n = templates.shape
    CM = np.zeros((K, K), float)
    rng_loc = np.random.default_rng(seed)

    for true_idx in range(K):
        t_true = templates[true_idx]
        counts = np.zeros(K, dtype=np.int64)
        left = N
        while left > 0:
            m = min(batch, left)
            flips = rng_loc.random((m, n)) < p_flip
            X = np.bitwise_xor(t_true, flips.astype(np.int8))  # (m, n)

            scores = np.empty((m, K), dtype=float)
            for i in range(K):
                t = templates[i]
                L10 = np.sum((X == 1) & (t == 0), axis=1)
                P01 = np.sum((X == 0) & (t == 1), axis=1)
                d = L10 + P01
                base = d * np.log((1 - p_flip) / p_flip) + (n - d) * np.log(p_flip / (1 - p_flip))
                scores[:, i] = base + np.log(priors[i])

            preds = np.argmax(scores, axis=1)
            for j in range(K):
                counts[j] += np.count_nonzero(preds == j)
            left -= m

        CM[true_idx] = counts / counts.sum()

    return CM

# РИСОВАНИЕ СЕТКИ 2×3 (теория/эксперимент × 3 случая приоров)
def plot_priors_grid(templates, pI, cases, N_exp, N_theor, class_names):

    K = templates.shape[0]
    mats_theor, mats_exp = [], []

    # считаем заранее (чтобы общий color scale подобрать честно)
    for _, priors in cases:
        CMt = theoretical_confusion_matrix(templates, priors, pI, N=N_theor)
        CMe = experimental_confusion_matrix(templates, priors, pI, N_exp, rng)
        mats_theor.append(CMt); mats_exp.append(CMe)

    vmin, vmax = 0.0, max(np.max(m) for m in (mats_theor + mats_exp))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(f"Влияние априорных вероятностей | pI = {pI}", fontsize=16)

    # верхний ряд — теоретические
    for j, ((title, pri), CM) in enumerate(zip(cases, mats_theor)):
        ax = axes[0, j]
        im = ax.imshow(CM, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\npriors={np.round(pri,3)}\nТеоретическая", fontsize=11)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
        for i in range(K):
            for k in range(K):
                ax.text(k, i, f"{CM[i,k]:.2f}", ha="center", va="center", color="w", fontsize=10)

    # нижний ряд — экспериментальные
    for j, ((title, pri), CM) in enumerate(zip(cases, mats_exp)):
        ax = axes[1, j]
        im = ax.imshow(CM, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\npriors={np.round(pri,3)}\nЭкспериментальная", fontsize=11)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
        for i in range(K):
            for k in range(K):
                ax.text(k, i, f"{CM[i,k]:.2f}", ha="center", va="center", color="w", fontsize=10)

    # общий colorbar справа, чтобы ничего не "съезжало"
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.subplots_adjust(right=0.9, wspace=0.3, hspace=0.35)
    plt.show()

# ЗАПУСК
if __name__ == "__main__":

    # Показать черно-белые изображения букв P/M/I
    plot_letters_bw([
        ("P", P_8x8),
        ("M", M_8x8),
        ("I", I_8x8),
    ], save=False)

    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    titles = ["P", "M", "I"]
    for i, (ax, mat, t) in enumerate(zip(axs, [P_d, M_d, I_d], titles)):
        ax.imshow(mat, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"Искажённая {t}")
        ax.axis("off")



    plt.tight_layout()
    plt.show()


    # ТРИ СЛУЧАЯ АПРИОРОВ
    cases = [
        ("Случай 1: p(w1)>p(w2)", np.array([0.7, 0.2, 0.1], float)),
        ("Случай 2: равные p(w)",       np.array([1/3, 1/3, 1/3], float)),
        ("Случай 3: p(w1)<p(w2)", np.array([0.1, 0.2, 0.7], float)),
    ]

    # Вывод 2×3 теплокарт (теория/эксперимент × 3 случая)
    plot_priors_grid(
        templates=templates,
        pI=pI,
        cases=cases,
        N_exp=N_TRIALS_PER_CLASS,
        N_theor=THEOR_N,
        class_names=class_names
    )

    # При необходимости — печать матриц в консоль для среднего (равного) случая:
    pri_equal = np.array([1/3, 1/3, 1/3], float)
    CM_theor_eq = theoretical_confusion_matrix(templates, pri_equal, pI, N=THEOR_N)
    CM_exp_eq   = experimental_confusion_matrix(templates, pri_equal, pI, N_TRIALS_PER_CLASS, rng)

    np.set_printoptions(precision=3, suppress=True)
