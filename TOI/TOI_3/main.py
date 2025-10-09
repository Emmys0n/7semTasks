# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Ellipse

# ==========================
# Вариант №1 (ЛР-3, разные ковариации)
# ==========================
m1 = np.array([ 2.0,  1.0])  # центр класса 1
m2 = np.array([-1.0,  1.0])  # центр класса 2

C1 = np.array([[3.0, -1.0],
               [-1.0, 3.0]])  # ковариация класса 1
C2 = np.array([[5.0,  2.0],
               [2.0,  6.0]])  # ковариация класса 2

p1 = 0.5  # априорная вероятность класса 1
p2 = 0.5  # априорная вероятность класса 2

np.random.seed(42)

# Объекты многомерных нормальных распределений
rv1 = multivariate_normal(mean=m1, cov=C1)
rv2 = multivariate_normal(mean=m2, cov=C2)

# ==========================
# Байесовский классификатор (QDA)
# ==========================
def bayes_classifier(x, rv1, rv2, p1, p2):
    # Сравниваем P(w1|x) ~ p(x|w1)P(w1) и P(w2|x) ~ p(x|w2)P(w2)
    # Можно работать в pdf (вероятностях), SciPy сам стабилен для 2D
    return 1 if (rv1.pdf(x) * p1) >= (rv2.pdf(x) * p2) else 2

# ==========================
# Теоретическая (MC) и экспериментальная СУММАРНЫЕ ошибки 1-го рода
# ==========================
def theoretical_total_error_type1_class1(rv1, rv2, p1, p2, sample_size, mc_n=100000):
    """
    MC-оценка теоретической вероятности ошибки 1-го рода:
    P(решить класс 2 | истинный класс 1).
    Возвращаем ожидаемое КОЛИЧЕСТВО ошибок для заданного sample_size.
    """
    samples = rv1.rvs(size=mc_n)
    # Векторизованно классифицируем
    prob1 = rv1.pdf(samples) * p1
    prob2 = rv2.pdf(samples) * p2
    mis = (prob2 > prob1).sum()
    prob = mis / mc_n
    return prob * sample_size  # ожидаемое число ошибок

def experimental_total_error_type1_class1(rv1, rv2, p1, p2, sample_size, num_trials=50):
    """
    Эксперимент: генерируем выборки из класса 1, считаем КОЛИЧЕСТВО ошибок (1-го рода) в каждой,
    возвращаем среднее и СКО по повторам.
    """
    totals = []
    for _ in range(num_trials):
        samples = rv1.rvs(size=sample_size)
        # Векторизованная классификация
        prob1 = rv1.pdf(samples) * p1
        prob2 = rv2.pdf(samples) * p2
        err_cnt = (prob2 > prob1).sum()
        totals.append(err_cnt)
    totals = np.array(totals, dtype=float)
    return totals.mean(), totals.std(ddof=1)

# ==========================
# Визуализация в стиле друга: 3 subplot'а
# ==========================
def visualize_original_data(rv1, rv2, p1, p2, num_samples=1000):
    # 1) Генерация тестовых данных (по 1000 из каждого класса)
    samples1 = rv1.rvs(size=num_samples)
    samples2 = rv2.rvs(size=num_samples)
    X = np.vstack([samples1, samples2])
    y_true = np.array([1]*num_samples + [2]*num_samples)

    # Классификация (векторизованно)
    prob1 = rv1.pdf(X) * p1
    prob2 = rv2.pdf(X) * p2
    y_pred = np.where(prob1 >= prob2, 1, 2)

    # Матрица ошибок и точность
    cm = confusion_matrix(y_true, y_pred, labels=[1,2])
    acc = np.trace(cm) / np.sum(cm)

    # Общая фигура
    plt.figure(figsize=(15, 5))

    # (1) Матрица ошибок
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Класс 1','Класс 2'],
                yticklabels=['Класс 1','Класс 2'])
    plt.title(f'Матрица ошибок\nТочность: {acc:.3f}')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')

    # Диапазоны для осей — по данным
    all_x = X[:,0]; all_y = X[:,1]
    x_min, x_max = all_x.min()-2, all_x.max()+2
    y_min, y_max = all_y.min()-2, all_y.max()+2

    # (2) Облака точек + эллипсы
    plt.subplot(1, 3, 2)
    plt.scatter(samples1[:,0], samples1[:,1], s=10, alpha=0.6, label='Класс 1', color='red')
    plt.scatter(samples2[:,0], samples2[:,1], s=10, alpha=0.6, label='Класс 2', color='blue')
    # центры
    plt.scatter(m1[0], m1[1], color='darkred', s=100, marker='x', linewidth=3, label='Центр 1')
    plt.scatter(m2[0], m2[1], color='darkblue', s=100, marker='x', linewidth=3, label='Центр 2')
    # эллипсы 95%
    for mean, cov, color in zip([m1, m2], [C1, C2], ['red', 'blue']):
        eigenvals, eigenvecs = np.linalg.eigh(cov)  # eigenvalues asc
        # χ²(0.95; df=2) ≈ 5.991
        width  = 2*np.sqrt(5.991*eigenvals[1])
        height = 2*np.sqrt(5.991*eigenvals[0])
        angle = np.degrees(np.arctan2(eigenvecs[1,1], eigenvecs[0,1]))
        plt.gca().add_patch(Ellipse(xy=mean, width=width, height=height,
                                    angle=angle, alpha=0.3, color=color, label=f'Эллипс 95%'))
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
    plt.xlabel('Признак 1 (X)')
    plt.ylabel('Признак 2 (Y)')
    plt.title('Распределения и эллипсы ковариации')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # (3) Разделяющая граница (QDA)
    plt.subplot(1, 3, 3)
    xs = np.linspace(x_min, x_max, 300)
    ys = np.linspace(y_min, y_max, 300)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

    # Решаю по байесовскому правилу
    P1_grid = rv1.pdf(grid) * p1
    P2_grid = rv2.pdf(grid) * p2
    Z = np.where(P1_grid >= P2_grid, 1, 2).reshape(XX.shape)

    # Граница по уровню 1.5
    plt.contour(XX, YY, Z, levels=[1.5], colors='black', linestyles='dashed', linewidths=2)
    # Поля
    plt.contourf(XX, YY, Z, levels=[0.5, 1.5, 2.5], colors=['red', 'blue'], alpha=0.15)

    # Центры и эллипсы добавим для наглядности
    plt.scatter(m1[0], m1[1], color='darkred', s=100, marker='x', linewidth=3, label='Центр 1')
    plt.scatter(m2[0], m2[1], color='darkblue', s=100, marker='x', linewidth=3, label='Центр 2')
    for mean, cov, color in zip([m1, m2], [C1, C2], ['red', 'blue']):
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        width  = 2*np.sqrt(5.991*eigenvals[1])
        height = 2*np.sqrt(5.991*eigenvals[0])
        angle = np.degrees(np.arctan2(eigenvecs[1,1], eigenvecs[0,1]))
        plt.gca().add_patch(Ellipse(xy=mean, width=width, height=height,
                                    angle=angle, alpha=0.25, color=color))
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
    plt.xlabel('Признак 1 (X)')
    plt.ylabel('Признак 2 (Y)')
    plt.title('Разделяющая граница (QDA)')
    plt.grid(True, alpha=0.3)

    plt.suptitle('Визуализация исходных данных и классификации', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return cm

# ==========================
# Запуск визуализации + исследовательская часть
# ==========================
if __name__ == "__main__":
    cm = visualize_original_data(rv1, rv2, p1, p2)

    # Размеры выборок (как у друга)
    sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    differences = []
    difference_stds = []

    print("Объем выборки | Разность (эксп - теор) | СКО эксперимента")
    print("-" * 60)

    for size in sample_sizes:
        exp_total_error, exp_std = experimental_total_error_type1_class1(rv1, rv2, p1, p2, size, num_trials=50)
        theor_total_error = theoretical_total_error_type1_class1(rv1, rv2, p1, p2, size, mc_n=100000)

        diff = exp_total_error - theor_total_error
        differences.append(diff)
        difference_stds.append(exp_std)

        print(f"{size:13d} | {diff:+8.2f}               | {exp_std:6.2f}")

    # График разности суммарных ошибок (эксп - теор) c СКО
    plt.figure(figsize=(12, 8))
    plt.errorbar(sample_sizes, differences, yerr=difference_stds,
                 fmt='o-', linewidth=2, markersize=6, capsize=6, capthick=2,
                 label='Разность суммарных ошибок (эксп − теор) ± СКО', color='green', alpha=0.8)
    plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Идеальное совпадение')
    plt.xscale('log')
    plt.xlabel('Объем выборки (число испытаний)', fontsize=12)
    plt.ylabel('Разность суммарных ошибок 1-го рода', fontsize=12)
    plt.title('Разность суммарной экспериментальной и теоретической\nошибок первого рода (класс 1) от объема выборки', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
