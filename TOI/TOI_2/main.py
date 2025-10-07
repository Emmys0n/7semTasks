import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Ellipse

# Исходные данные
m1 = np.array([2, -1])   # Математическое ожидание (центр) класса 1
m2 = np.array([-1, 1])   # Математическое ожидание (центр) класса 2
C = np.array([[3, -1],   # Общая ковариационная матрица для всех классов
              [-1, 3]])

# Функция для вычисления дискриминантной функции для одного класса
def discriminant(x, m, C_inv, P):
    # g(x) = x^T * C^{-1} * m - 0.5 * m^T * C^{-1} * m + ln(P)
    return x @ C_inv @ m - 0.5 * m @ C_inv @ m + np.log(P)

# Функция классификации точки x по двум классам
def classify(x, means, C_inv, priors):
    discriminants = []
    for i, m in enumerate(means):
        d = discriminant(x, m, C_inv, priors[i])
        discriminants.append(d)
    return np.argmax(discriminants) + 1  # возвращает класс 1 или 2

# Основная функция эксперимента
def run_experiment(means, cov, priors, title, num_samples=1000):
    C_inv = np.linalg.inv(cov)

    # Генерация выборок
    samples = []
    y_true = []
    for i, m in enumerate(means):
        sample = np.random.multivariate_normal(m, cov, num_samples)
        samples.append(sample)
        y_true.extend([i + 1] * num_samples)

    X = np.vstack(samples)
    y_pred = [classify(x, means, C_inv, priors) for x in X]

    # Матрица ошибок и точность
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.trace(cm) / np.sum(cm)

    # Визуализация
    plt.figure(figsize=(15, 5))

    # 1. Матрица ошибок
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Матрица ошибок\nТочность: {accuracy:.3f}')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')

    # 2. Распределение точек и разделяющая граница
    plt.subplot(1, 3, 2)
    colors = ['red', 'blue']
    for i, sample in enumerate(samples):
        plt.scatter(sample[:, 0], sample[:, 1], alpha=0.6, label=f'Класс {i + 1}', color=colors[i], s=10)

    # Построение разделяющей линии
    x_vals = np.linspace(-5, 5, 200)
    y_vals = np.linspace(-5, 5, 200)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X_grid)
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            point = np.array([X_grid[i, j], Y_grid[i, j]])
            Z[i, j] = classify(point, means, C_inv, priors)
    plt.contour(X_grid, Y_grid, Z, levels=[1.5], colors='black', linestyles='dashed', linewidths=1)
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title('Распределение классов')
    plt.legend()
    plt.grid(True)

    # 3. Эллипсы ковариации
    plt.subplot(1, 3, 3)
    for i, m in enumerate(means):
        eigenvals, eigenvecs = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = 2 * np.sqrt(5.991 * eigenvals[0])
        height = 2 * np.sqrt(5.991 * eigenvals[1])
        ellipse = Ellipse(xy=m, width=width, height=height, angle=angle, alpha=0.3, color=colors[i])
        plt.gca().add_patch(ellipse)
        plt.scatter(m[0], m[1], c=colors[i], s=80, marker='x', label=f'Центр {i + 1}')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title('Форма областей классов')
    plt.legend()
    plt.grid(True)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return cm, accuracy

# Основная программа
if __name__ == "__main__":
    priors = [0.5, 0.5]

    cm_original, acc_original = run_experiment([m1, m2], C, priors, "Исходные данные (вариант 21)")

    # a) Увеличить вероятность правильного распознавания
    m1_a = np.array([4, -3])
    m2_a = np.array([-3, 3])
    cm_a, acc_a = run_experiment([m1_a, m2_a], C, priors, "a) Увеличение расстояния между классами")

    # b) Увеличить суммарную ошибку
    m1_b = np.array([0, 0])
    m2_b = np.array([0.5, 0.5])
    cm_b, acc_b = run_experiment([m1_b, m2_b], C, priors, "b) Приближение центров классов")

    # c) Увеличить ошибку 1-го рода, уменьшить ошибку 2-го рода
    priors_c = [0.3, 0.7]
    m1_c = np.array([1, -0.5])
    m2_c = np.array([-1, 1])
    cm_c, acc_c = run_experiment([m1_c, m2_c], C, priors_c, "c) Смещение вероятностей в пользу класса 2")

    # d) Увеличить ошибку 2-го рода, уменьшить ошибку 1-го рода
    priors_d = [0.7, 0.3]
    m1_d = np.array([2, -1])
    m2_d = np.array([-2, 1])
    cm_d, acc_d = run_experiment([m1_d, m2_d], C, priors_d, "d) Смещение вероятностей в пользу класса 1")

    # e) Растяжение по оси X
    C_e = np.array([[9, -1],
                    [-1, 3]])
    cm_e, acc_e = run_experiment([m1, m2], C_e, priors, "e) Растяжение по горизонтали")

    # f) Зеркальное отражение
    C_f = np.array([[3, 1],
                    [1, 3]])
    m1_f = np.array([-m1[0], m1[1]])
    m2_f = np.array([-m2[0], m2[1]])
    cm_f, acc_f = run_experiment([m1_f, m2_f], C_f, priors, "f) Зеркальное отражение форм кластеров")

    print("=" * 60)
    print("Сводная таблица точностей")
    print("=" * 60)
    results = {
        'Исходные': acc_original,
        'a) Улучшение': acc_a,
        'b) Ухудшение': acc_b,
        'c) Ошибка 1↑ 2↓': acc_c,
        'd) Ошибка 1↓ 2↑': acc_d,
        'e) Растяжение': acc_e,
        'f) Отражение': acc_f
    }
    for name, acc in results.items():
        change = acc - acc_original
        print(f"{name:<15} | Точность: {acc:.3f} | Изменение: {change:+.3f}")
