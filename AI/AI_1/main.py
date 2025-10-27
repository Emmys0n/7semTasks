import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import MDS

# ========= 1) Загрузка данных (устойчиво к наличию/отсутствию заголовка) =========
path = "French Food Data.dat"

try:
    df_try = pd.read_csv(path, sep=r"\s+", engine="python")
    has_all_cols = {"n_", "type_", "bread", "vegetables", "fruits", "meat", "poultry", "milk", "wine"}.issubset(df_try.columns)
    if has_all_cols:
        df = df_try
    else:
        columns = ["n_", "type_", "bread", "vegetables", "fruits", "meat", "poultry", "milk", "wine"]
        df = pd.read_csv(path, sep=r"\s+", header=None, names=columns, engine="python")
except Exception:
    columns = ["n_", "type_", "bread", "vegetables", "fruits", "meat", "poultry", "milk", "wine"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns, engine="python")

print("Fremch Food Data:")
print(df.to_string(index=False))   # выведет все строки нашего сэта

# ========= 2) Отбор признаков и стандартизация =========
features = ["bread", "vegetables", "fruits", "meat", "poultry", "milk", "wine"]
X = df[features].astype(float).values  # важно привести к float

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========= 3) Дендрограмма (иерархическая кластеризация) =========
plt.figure(figsize=(10, 6))
labels = None
# Попробуем сделать понятные подписи: type_ + номер наблюдения
try:
    labels = (df["type_"].astype(str) + "_n" + df["n_"].astype(str)).tolist()
except Exception:
    labels = None  # если не получилось — пусть будут индексы

sch.dendrogram(sch.linkage(X_scaled, method="ward"), labels=labels)
plt.title("Дендрограмма (иерархическая кластеризация, Ward)")
plt.xlabel("Наблюдения")
plt.ylabel("Евклидово расстояние")
plt.tight_layout()
plt.show()

# Выбор числа кластеров для HC (можешь потом поменять)
n_hc = 3
hc = AgglomerativeClustering(n_clusters=n_hc, metric="euclidean", linkage="ward")
df["Cluster_HC"] = hc.fit_predict(X_scaled)

# ========= 4) Метод локтя для K-means =========
inertia = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(K_range), inertia, marker="o")
plt.title("Метод локтя (K-means)")
plt.xlabel("Число кластеров k")
plt.ylabel("Инерция (within-cluster SSE)")
plt.tight_layout()
plt.show()

# ========= 5) K-means с выбранным 3 k  =========
k_opt = 3
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

# ========= 6) Визуализация кластеров через MDS =========
mds = MDS(n_components=2, random_state=42, dissimilarity="euclidean")
X_mds = mds.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_mds[:, 0], X_mds[:, 1], c=df["Cluster_KMeans"], cmap="tab10", s=90, edgecolors="k")
# подпишем точки
for i in range(len(df)):
    label = df["type_"].astype(str).iloc[i] if "type_" in df.columns else str(i)
    plt.text(X_mds[i, 0] + 0.02, X_mds[i, 1] + 0.02, label, fontsize=9)
plt.title("MDS визуализация кластеров (K-means)")
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.tight_layout()
plt.show()

# ========= 7) Сводки для интерпретации =========
print("\nСредние значения признаков по кластерам (K-means):")
print(df.groupby("Cluster_KMeans")[features].mean().round(2))

print("\nРаспределение типов семей по кластерам (K-means):")
if "type_" in df.columns:
    print(df.groupby(["Cluster_KMeans", "type_"]).size())
else:
    print("Колонки type_ нет — пропускаю сводку по типам.")

print("\nСредние значения признаков по кластерам (HC):")
print(df.groupby("Cluster_HC")[features].mean().round(2))
