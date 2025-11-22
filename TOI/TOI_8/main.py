import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ЗАДАНИЕ ВАРИАНТА
# коэффициенты полинома (a(1)...a(6))
a1, a2, a3, a4, a5, a6 = 2, -3, 17, 300, 250, -1100
a = np.array([a1, a2, a3, a4, a5, a6], dtype=float)

ng = 5          # порядок полинома
sigma2 = 1.0    # дисперсия шума (D = 1)
gamma = 0.01    # уровень значимости (alpha = 0.01)


def true_poly(x: np.ndarray) -> np.ndarray:
    """Истинная полиномиальная функция."""
    return (a[0] * x**5 +
            a[1] * x**4 +
            a[2] * x**3 +
            a[3] * x**2 +
            a[4] * x +
            a[5])


def run_experiment(N: int, random_state: int = 42):
    """
    Строим регрессию для заданного объёма выборки N.
    Рисуем график и печатаем характеристики.
    """
    rng = np.random.default_rng(random_state)

    # 1) Генерация выборки
    # x ~ U[0, 1]
    x = rng.uniform(0.0, 1.0, size=N)

    # истинное значение и зашумлённые наблюдения
    y_true = true_poly(x)
    noise = rng.normal(0.0, np.sqrt(sigma2), size=N)
    y = y_true + noise

    # 2) Матрица признаков X: [x^5, x^4, x^3, x^2, x, 1]
    X = np.column_stack([x**5, x**4, x**3, x**2, x, np.ones_like(x)])

    # 3) Линейная регрессия OLS (метод наименьших квадратов)
    model = sm.OLS(y, X)
    res = model.fit()

    # 4) Предсказания и доверительные интервалы на сетке
    x_grid = np.linspace(0.0, 1.0, 400)
    X_grid = np.column_stack(
        [x_grid**5, x_grid**4, x_grid**3, x_grid**2, x_grid, np.ones_like(x_grid)]
    )

    y_true_grid = true_poly(x_grid)

    pred = res.get_prediction(X_grid)
    # доверительный интервал уровня (1 - gamma), т.е. 99%
    frame = pred.summary_frame(alpha=gamma)
    y_pred_grid = frame["mean"]
    ci_lower = frame["mean_ci_lower"]
    ci_upper = frame["mean_ci_upper"]

    # 5) «Дисперсии оценивания»
    # ошибка между оценённой моделью и истинной функцией
    sigma2_reg = np.mean((y_pred_grid - y_true_grid) ** 2)

    # ошибка между моделью и наблюдениями (станд. MSE на обучающей выборке)
    y_hat_train = res.fittedvalues
    sigma2_lms = np.mean((y_hat_train - y) ** 2)

    beta = sigma2_reg / sigma2_lms if sigma2_lms > 0 else np.nan

    # 6) Печать результатов (аналог командного окна MATLAB)
    print(f"\nN = {N}")
    print(f"betta = {beta:.4f}")
    print("Дисперсия оценивания (погрешности модели) "
          "множественной линейной регрессии:",
          f"{sigma2_reg:.4f}")
    print("Дисперсия оценивания (погрешности модели) "
          "метода наименьших квадратов:",
          f"{sigma2_lms:.4f}")
    print(f"Коэффициент детерминации R^2 = {res.rsquared:.6f}")
    # Критерий фишера
    print(f"F-статистика = {res.fvalue:.4f}, p-value = {res.f_pvalue:.3g}")

    # 7) График как в MATLAB
    plt.figure(figsize=(7, 6))
    plt.title("Полученная регрессионная зависимость")

    # красные кружки – наблюдения XN-YN
    plt.scatter(x, y, facecolors='none', edgecolors='r',
                label="XN-YN (наблюдения)")

    # синяя линия – истинная функция y = f(x)
    plt.plot(x_grid, y_true_grid, 'b', label="y = f(x) (истинная функция)")

    # зелёная линия – оценка регрессии y = f_lms(x)
    plt.plot(x_grid, y_pred_grid, 'g', label="y = f_лмс(x) (регрессия OLS)")

    # пунктир – доверительный интервал y ± dy
    plt.plot(x_grid, ci_lower, 'k--', label="y - dy (доверит. интервал)")
    plt.plot(x_grid, ci_upper, 'k--', label="y + dy")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)

    # подпись в стиле MATLAB: N, ng, D, gamma, p-val
    txt = f"N={N} ng={ng} D={sigma2:.0f} gamma={gamma:.2f} p-val={res.f_pvalue:.3g}"
    plt.text(0.02, 0.95, txt,
             transform=plt.gca().transAxes,
             fontsize=9,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.legend()
    plt.tight_layout()
    plt.show()

    return res


#  ЗАПУСК ДЛЯ РАЗНЫХ ОБЪЁМОВ ВЫБОРКИ 
for N in [50, 100, 1000]:
    run_experiment(N)
