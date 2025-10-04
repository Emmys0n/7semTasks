import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# === 1. Загрузка данных ===
file_path = "beer2.csv"
data = pd.read_csv(file_path)

# временной индекс (t = 1..N)
data['t'] = np.arange(1, len(data) + 1)

# === 2. Построим график исходного временного ряда ===
plt.figure(figsize=(10,5))
plt.plot(data['t'], data['beer'], marker='o')
plt.title("Australian Beer Production (1991-1995)")
plt.xlabel("Месяц (t)")
plt.ylabel("Производство пива")
plt.grid(True)
plt.show()

# === 3. Линейная регрессионная модель ===
X = data[['t']]
y = data['beer']

model = LinearRegression()
model.fit(X, y)

# === 4. Прогноз на 8 месяцев вперёд ===
future_t = np.arange(len(data) + 1, len(data) + 9).reshape(-1, 1)
forecast = model.predict(future_t)

# === 5. Подготовим данные для визуализации ===
full_t = np.concatenate([data['t'], future_t.flatten()])
full_pred = np.concatenate([model.predict(X), forecast])

plt.figure(figsize=(12,6))
plt.plot(data['t'], data['beer'], label="Фактические данные", marker='o')
plt.plot(full_t, full_pred, label="Линейная регрессия + прогноз", color="red")
plt.axvline(x=len(data), color="gray", linestyle="--", label="Начало прогноза")
plt.title("Линейный регрессионный анализ временного ряда")
plt.xlabel("Месяц (t)")
plt.ylabel("Производство пива")
plt.legend()
plt.grid(True)
plt.show()

# === 6. Вывод спрогнозированных значений ===
print("Прогноз на 8 месяцев вперёд:")
for i, val in enumerate(forecast, start=1):
    print(f"Месяц {i}: {val:.2f}")
