import matplotlib.pyplot as plt
import numpy as np
import math

# Параметры распределения
b = 2.2
c = 6.2

# Теоретические значения
m_theory = c * b # Мат. Ожидание
D_theory = c * b**2 # Дисперсия

print(f"Теоретическое мат. ожидание: {m_theory:.3f}")
print(f"Теоретическая дисперсия: {D_theory:.3f}")

# Генерация одной случайной величины гамма-распределения
def generate_gamma(b, c):
    return np.random.gamma(shape=c, scale=b)  # Форма, мастшаб

# Плотность гамма-распределения в точке x
def gamma_pdf(x, b, c):
    # Формула плотности: f(x) = (x^(c - 1) * exp(-x / b)) / (b^c * Γ(c))
    # x - точка, в которой вычисляется плотность
    # b - параметр масштаба, c - параметр формы
    if x < 0:
        return 0.0
    return (x**(c-1) * np.exp(-x/b)) / (b**c * math.gamma(c))

# Генерация данных для анализа
np.random.seed(42)  # для воспроизводимости результатов
n_max = 30000 # объем выборки для анализа
n_values = np.arange(1, n_max + 1)

# Генерация всех значений сразу
all_samples = np.array([generate_gamma(b, c) for _ in range(n_max)])

# Расчет накапливающихся статистик
cumulative_mean = np.cumsum(all_samples) / n_values                                           # выборочное мат. ожидание (среднее)
cumulative_var  = np.cumsum((all_samples - cumulative_mean)**2) / np.maximum(n_values - 1, 1) # выборочная дисперсия как функция от числа реализаций

# Задание 1: График выборочного мат. ожидания
plt.figure(figsize=(10, 6))
plt.plot(n_values, cumulative_mean, label='Выборочное мат. ожидание')
plt.axhline(y=m_theory, color='r', linestyle='--', label='Теоретическое мат. ожидание')
plt.xlabel('Число реализаций')
plt.ylabel('Мат. ожидание')
plt.title('Зависимость выборочного мат. ожидания от числа реализаций (гамма-распределение)')
plt.legend()
plt.grid(True)
plt.show()

# Задание 2: График выборочной дисперсии
plt.figure(figsize=(10, 6))
plt.plot(n_values, cumulative_var, label='Выборочная дисперсия')
plt.axhline(y=D_theory, color='r', linestyle='--', label='Теоретическая дисперсия')
plt.xlabel('Число реализаций')
plt.ylabel('Дисперсия')
plt.title('Зависимость выборочной дисперсии от числа реализаций (гамма-распределение)')
plt.legend()
plt.grid(True)
plt.show()

# Задание 3: Гистограмма и плотность распределения
plt.figure(figsize=(10, 6))
n_bins = 50
counts, bin_edges, _ = plt.hist(all_samples, bins=n_bins, density=True,
                               alpha=0.7, label='Эмпирическая плотность')

# Теоретическая плотность
x = np.linspace(0, max(all_samples), 1000) # 1000 точек от 0 до максимума выборки
pdf_values = [gamma_pdf(x_i, b, c) for x_i in x] # вычисление плотности в каждой точке
plt.plot(x, pdf_values, 'r-', linewidth=2, label='Теоретическая плотность')

plt.xlabel('Значение СВ')
plt.ylabel('Плотность вероятности')
plt.title('Эмпирическая и теоретическая плотности гамма-распределения')
plt.legend()
plt.grid(True)
plt.show()

# Расчет средней абсолютной разности эмпирических и теоретических значений
def calculate_mean_abs_diff(samples, b, c, n_bins=50):
    # samples - массив случайных величин
    # b, c - параметры гамма-распределения
    # n_bins - количество интервалов гистограммы
    hist, bin_edges = np.histogram(samples, bins=n_bins, density=True) # гистограмма с нормировкой
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 # центры интервалов гистограммы
    theoretical_pdf = np.array([gamma_pdf(x, b, c) for x in bin_centers]) # теоретическая плотность в центрах
    return np.mean(np.abs(hist - theoretical_pdf)) # средняя абсолютная разность (MAD)

# График зависимости средней абсолютной разности
mad_values = [] # список для хранения значений средней абсолютной разности
for n in range(100, n_max + 1, 100):
    mad = calculate_mean_abs_diff(all_samples[:n], b, c) # берём первые n элементов
    mad_values.append(mad)

plt.figure(figsize=(10, 6))
plt.plot(range(100, n_max + 1, 100), mad_values)
plt.xlabel('Число реализаций')
plt.ylabel('Средняя абсолютная разность')
plt.title('Зависимость точности оценки плотности от числа реализаций (гамма-распределение)')
plt.grid(True)
plt.show()
