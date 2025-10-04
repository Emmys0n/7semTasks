# -*- coding: utf-8 -*-
# kNN классификация для Car_Data.dat (разделитель ';', первая колонка MODEL)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

# 0) Загрузка и подготовка
path = "Car_Data.dat"  # поправь путь при необходимости

# В файле заголовок: MODEL;P;M;R78;R77;H;R;TR;W;L;T;D;G;C
df = pd.read_csv(
    path,
    sep=";",
    header=0,
    na_values=["NA", "NaN", ""],
    engine="python"
)

# Переименуем столбцы в осмысленные имена и уберём MODEL
col_map = {
    "P": "price",
    "M": "mileage",
    "R78": "repair_78",
    "R77": "repair_77",
    "H": "headroom",
    "R": "rear_seat",
    "TR": "trunk",
    "W": "weight",
    "L": "length",
    "T": "turn_diam",
    "D": "displacement",
    "G": "gear_ratio",
    "C": "company",
}
if "MODEL" in df.columns:
    df = df.drop(columns=["MODEL"])
df = df.rename(columns=col_map)

assert df.shape[1] == 13, f"Ожидалось 13 колонок после подготовки, получено: {df.shape[1]}"
df["company"] = df["company"].astype(int)

print("Loaded shape:", df.shape)
print(df.head())

# 1) X/y и train/test
feature_names = df.drop(columns=["company"]).columns.to_numpy()
X = df.drop(columns=["company"]).values
y = df["company"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 2) Пайплайн: имьютация -> стандартизация -> kNN
base_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# 3) Подбор k/метрики/весов по 5-fold CV
param_grid = {
    "knn__n_neighbors": list(range(1, 21)),
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    base_pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_model   = grid.best_estimator_
best_params  = grid.best_params_
cv_best_score= grid.best_score_

print("\nЛучшие параметры (CV):", best_params)
print(f"Лучший accuracy по CV: {cv_best_score:.3f}")

# 4) Оценка на тестовой
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
err_pct = (1 - acc) * 100

print("\n=== Тест ===")
print(f"Accuracy: {acc:.3f}  |  Ошибки: {err_pct:.2f}%")
print("\nConfusion matrix (истина по строкам, предсказание по столбцам):")
print(confusion_matrix(y_test, y_pred))
print("\nКлассификационный отчёт:")
print(classification_report(y_test, y_pred, digits=3))

# 4.1) Кривая подбора k (фиксируем лучшие weights/metric)
fixed_metric  = best_params["knn__metric"]
fixed_weights = best_params["knn__weights"]

k_list = list(range(1, 21))
acc_list = []

for k in k_list:
    pipe_k = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights=fixed_weights, metric=fixed_metric))
    ])
    scores = []
    for tr_idx, val_idx in cv.split(X_train, y_train):
        pipe_k.fit(X_train[tr_idx], y_train[tr_idx])
        scores.append(pipe_k.score(X_train[val_idx], y_train[val_idx]))
    acc_list.append(np.mean(scores))

plt.figure(figsize=(8,5))
plt.plot(k_list, acc_list, marker="o")
plt.title("Подбор k (accuracy по 5-fold CV)")
plt.xlabel("k (число соседей)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# 5) (Опц.) Сокращение признаков без потери качества
# ВАЖНО: теперь порядок — ИМПУТЕР → SELECTKBEST → СКЕЙЛЕР → KNN
best_m = None
best_m_acc = -np.inf
best_m_model = None

for m in range(3, 13):
    pipe_fs = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),          # <-- сначала имьютация
        ("selector", SelectKBest(score_func=f_classif, k=m)),   # <-- потом селектор (без NaN)
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=best_params["knn__n_neighbors"],
            weights=best_params["knn__weights"],
            metric=best_params["knn__metric"]
        ))
    ])
    pipe_fs.fit(X_train, y_train)
    y_pred_m = pipe_fs.predict(X_test)
    acc_m = accuracy_score(y_test, y_pred_m)
    if acc_m > best_m_acc:
        best_m_acc = acc_m
        best_m = m
        best_m_model = pipe_fs

print(f"\nЛучшее число признаков: k={best_m} | Accuracy на тесте: {best_m_acc:.3f}")
if best_m_acc >= acc:
    print("➡️ Можно сократить признаки без потери качества (или лучше).")
else:
    print("➡️ Сокращение признаков ухудшает качество.")

# Какие признаки выбраны
selector = best_m_model.named_steps["selector"]
mask = selector.get_support()
selected_features = list(feature_names[mask])
print("Выбранные признаки:", selected_features)
