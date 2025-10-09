import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Ellipse

# Исходные данные (вариант №21)
m1 = np.array([ 2, -1])  # Математическое ожидание (центр) класса 1
m2 = np.array([-1,  1])  # Математическое ожидание (центр) класса 2
C = np.array([[3, -1],  # Общая ковариационная матрица для всех классов (2x2 матрица)
               [-1, 3]])

# Функция для вычисления дискриминантной функции (решающей функции) для одного класса
def discriminant(x, m, C_inv, P):
    # Вычисление дискриминантной функции по формуле ЛДА:
    # g(x) = x^T * C^{-1} * m - 0.5 * m^T * C^{-1} * m + ln(P)
    return x @ C_inv @ m - 0.5 * (m @ C_inv @ m) + np.log(P)  # @ - оператор матричного умножения

# Функция классификации точки x по двум классам
def classify(x, means, C_inv, priors):
    discriminants = []                             # Список значений дискриминантов для каждого класса
    for i, m in enumerate(means):                  # i - индекс класса, m - центр класса
        d = discriminant(x, m, C_inv, priors[i])   # Вычисляем дискриминант для текущего класса
        discriminants.append(d)                    # Добавляем вычисленное значение в список
    return np.argmax(discriminants) + 1            # Возвращаем номер класса с максимальным дискриминантом (1 или 2)

# Основная функция проведения эксперимента с заданными параметрами
def run_experiment(means, cov, priors, title, num_samples=1000):
    C_inv = np.linalg.inv(cov)                     # Вычисляем обратную матрицу ковариаций (нужна для дискриминанта)

    # Генерация тестовых данных для каждого класса
    samples = []                                   # Здесь будем хранить массивы точек по классам
    y_true = []                                    # Истинные метки классов
    for i, m in enumerate(means):
        sample = np.random.multivariate_normal(m, cov, num_samples)  # Генерируем num_samples точек из N(m, C)
        samples.append(sample)                                   # Сохраняем точки класса i
        y_true.extend([i + 1] * num_samples)                     # Добавляем метки класса (i+1 повторяется num_samples раз)

    X = np.vstack(samples)                         # Объединяем все точки в одну матрицу размером (2*num_samples x 2)
    y_pred = [classify(x, means, C_inv, priors) for x in X]  # Классифицируем каждую точку и сохраняем предсказания

    # Вычисление матрицы ошибок и точности
    cm = confusion_matrix(y_true, y_pred)          # Строим матрицу ошибок (confusion matrix) 2x2
    accuracy = np.trace(cm) / np.sum(cm)           # Точность: сумма по диагонали / общее число объектов

    # Визуализация результатов (графики)
    plt.figure(figsize=(18, 5))                    # Общая фигура для трёх графиков

    # 1. График матрицы ошибок
    plt.subplot(1, 3, 1)                           # Первый подграфик
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Тепловая карта матрицы ошибок
    plt.title(f'Матрица ошибок\nТочность: {accuracy:.3f}')  # Заголовок с точностью
    plt.xlabel('Предсказанный класс')              # Подпись оси X
    plt.ylabel('Истинный класс')                   # Подпись оси Y

    # 2. График распределения точек и разделяющей поверхности
    plt.subplot(1, 3, 2)                           # Второй подграфик
    colors = ['red', 'blue']                       # Цвета для класса 1 и 2
    for i, sample in enumerate(samples):
        plt.scatter(sample[:, 0], sample[:, 1], alpha=0.6, label=f'Класс {i + 1}', color=colors[i], s=10)  # Точки классов

    # Построение решающей линии для двух классов (g1(x) = g2(x))
    x_vals = np.linspace(X[:,0].min()-2, X[:,0].max()+2, 250)  # Диапазон по оси X
    y_vals = np.linspace(X[:,1].min()-2, X[:,1].max()+2, 250)  # Диапазон по оси Y
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)                # Сетка для отрисовки границы
    Z = np.zeros_like(X_grid)                                   # Здесь будут классы для каждой точки сетки
    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            point = np.array([X_grid[i, j], Y_grid[i, j]])      # Создаем точку из координат сетки
            Z[i, j] = classify(point, means, C_inv, priors)     # Классифицируем точку сетки
    plt.contour(X_grid, Y_grid, Z, levels=[1.5], colors='black', linestyles='dashed', linewidths=1)  # Линия раздела

    # Отметим центры классов крестиками
    for i, m in enumerate(means):
        plt.scatter(m[0], m[1], color=colors[i], s=100, marker='x', linewidth=3, label=f'Центр {i + 1}')

    plt.xlabel('Признак 1')                        # Подпись оси X
    plt.ylabel('Признак 2')                        # Подпись оси Y
    plt.title('Распределение классов и граница')   # Заголовок графика
    plt.legend(loc='upper right', fontsize=8)      # Легенда
    plt.grid(True)                                 # Сетка

    # 3. График сравнения центров и форм распределений (эллипсы 95% ДИ)
    plt.subplot(1, 3, 3)                           # Третий подграфик
    for i, m in enumerate(means):                  # Для каждого класса рисуем эллипс
        eigenvals, eigenvecs = np.linalg.eig(cov)  # Собственные значения/векторы ковариационной матрицы
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))  # Угол поворота эллипса (в градусах)
        width  = 2 * np.sqrt(5.991 * eigenvals[0])  # Ширина эллипса (95% доверительный интервал)
        height = 2 * np.sqrt(5.991 * eigenvals[1])  # Высота эллипса (95% доверительный интервал)
        ellipse = Ellipse(xy=m, width=width, height=height, angle=angle, alpha=0.2, color=colors[i])  # Эллипс
        plt.gca().add_patch(ellipse)               # Добавляем эллипс на ось
        plt.scatter(m[0], m[1], color=colors[i], s=100, marker='o', linewidth=3, label=f'Центр {i + 1}')  # Точка-центр

    plt.xlabel('Признак 1')                        # Подпись оси X
    plt.ylabel('Признак 2')                        # Подпись оси Y
    plt.title('Сравнение центров и форм')          # Заголовок
    plt.legend(loc='upper right', fontsize=8)      # Легенда
    plt.grid(True)                                 # Сетка

    plt.suptitle(title, fontsize=14, fontweight='bold')  # Общий заголовок для фигуры
    plt.tight_layout()                             # Компоновка без наложений
    plt.show()                                     # Показать все три графика

    return cm, accuracy                            # Возвращаем матрицу ошибок и точность

if __name__ == "__main__":
    priors = [0.5, 0.5]  # Вероятности классов 1 и 2, априорные (равные)

    # Исходные данные
    cm_original, acc_original = run_experiment([m1, m2], C, priors, "Исходные данные")

    # a) Увеличить вероятность правильного распознавания
    m1_a = np.array([ 4, -3])  # Сдвигаем центр класса 1 дальше от класса 2
    m2_a = np.array([-3,  3])  # Сдвигаем центр класса 2 дальше от класса 1
    cm_a, acc_a = run_experiment([m1_a, m2_a], C, priors, "a) Увеличение расстояния между классами")

    # b) Увеличить суммарную ошибку
    m_mid = (m1 + m2) / 2                        # Геометрический центр между классами
    m1_b = m_mid + np.array([+0.2, -0.2])        # Приближаем центры друг к другу
    m2_b = m_mid + np.array([-0.2, +0.2])        # Чем ближе центры — тем больше перекрытие
    cm_b, acc_b = run_experiment([m1_b, m2_b], C, priors, "b) Приближение центров классов")

    # c) Увеличить ошибку 1-го рода, уменьшить ошибку 2-го рода (для класса 1)
    priors_c = [0.3, 0.7]
    m1_c = m1 + 0.5 * (m2 - m1)
    cm_c, acc_c = run_experiment([m1_c, m2], C, priors_c, "c) Ассиметричное изменение")

    # d) Увеличить ошибку 2-го рода, уменьшить ошибку 1-го рода (для класса 1)
    priors_d = [0.7, 0.3]
    m2_d = m2 + 0.5 * (m1 - m2)
    cm_d, acc_d = run_experiment([m1, m2_d], C, priors_d, "d) Обратная ассиметрия")

    # e) Увеличить протяженность кластеров в одном из направлений (растянуть форму по оси X)
    C_e = np.array([[9, -1],                      # Увеличиваем дисперсию по первому признаку (ось X)
                    [-1, 3]])                     # Дисперсия по второму признаку без изменений
    cm_e, acc_e = run_experiment([m1, m2], C_e, priors, "e) Растяжение по горизонтали")

    # f) Зеркально отразить форму областей локализации (по Y, меняем знак корреляции)
    C_f = np.array([[3,  1],                      # Меняем знак внедиагонального элемента (-1 -> +1)
                    [1,  3]])                     # Отражение по оси Y меняет наклон эллипсов
    m1_f = np.array([-m1[0], m1[1]])              # Координаты центров отражаем по X: (x, y) -> (-x, y)
    m2_f = np.array([-m2[0], m2[1]])
    cm_f, acc_f = run_experiment([m1_f, m2_f], C_f, priors, "f) Зеркальное отражение форм кластеров")

    print("=" * 60)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    results = {
        'Исходные': acc_original,
        'a) Улучшение': acc_a,
        'b) Ухудшение': acc_b,
        'c) Ассиметричное изменение': acc_c,
        'd) Обратная ассиметрия': acc_d,
        'e) Растяжение': acc_e,
        'f) Отражение': acc_f
    }

    for name, accuracy in results.items():
        change = accuracy - acc_original
        print(f"{name:<15} | Точность: {accuracy:.3f} | Изменение: {change:+.3f}")  # Форматированный вывод

    # Функция для анализа ошибок 1-го и 2-го рода (для 2-классового случая, класс 1 — «позитив»)
    def analyze_errors(cm, name):
        # Ошибка 1-го рода (False Negative для класса 1): объект класса 1 отнесён к классу 2
        error_type1 = cm[0, 1] / np.sum(cm[0, :])  # (ошибки класса 1) / (все объекты класса 1)

        # Ошибка 2-го рода (False Positive для класса 1): объект класса 2 отнесён к классу 1
        error_type2 = cm[1, 0] / np.sum(cm[1, :])  # (ложные принятия класса 1) / (все объекты класса 2)

        print(f"{name}:")                          # Выводим название эксперимента
        print(f"  Ошибка 1-го рода (класс 1): {error_type1:.3f}")  # Ошибка 1-го рода
        print(f"  Ошибка 2-го рода (класс 1): {error_type2:.3f}")  # Ошибка 2-го рода

    print("\n" + "=" * 60)
    print("АНАЛИЗ ОШИБОК 1-го И 2-го РОДА (класс 1)")
    print("=" * 60)
    analyze_errors(cm_original, "Исходные")        # Анализ исходных данных
    analyze_errors(cm_c, "c) Ошибка 1↑ 2↓")       # Анализ эксперимента c
    analyze_errors(cm_d, "d) Ошибка 1↓ 2↑")       # Анализ эксперимента d
