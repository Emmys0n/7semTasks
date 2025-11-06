# ai7_svm_report.py
# ЛР-7. Задание 3. SVM для 4 классов: (A) линейно разделимые и (B) с пересечением
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

RNG = np.random.default_rng(42)

# --
# Генерация данных (4 гауссовых класса)
# --
def generate_4_gaussians(n_per_class=300, dist=4.0, spread=0.25, rng=RNG):
    """
    4 класса в вершинах квадрата:
      0: (-dist, -dist), 1: (dist, -dist), 2: (dist, dist), 3: (-dist, dist)
    spread — множитель ковариационной матрицы I.
    """
    I = np.eye(2)
    cov = spread * I
    means = [(-dist, -dist), (dist, -dist), (dist, dist), (-dist, dist)]
    X_list, y_list = [], []
    for k, m in enumerate(means):
        Xk = rng.multivariate_normal(m, cov, n_per_class)
        yk = np.full(n_per_class, k, dtype=int)
        X_list.append(Xk)
        y_list.append(yk)
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y

# --
# Обучение и оценка
# --
def evaluate_svm(X, y, test_size=0.3, random_state=42, kfold_splits=5, kernel='linear'):
    clf = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0, decision_function_shape='ovr', random_state=random_state))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)

    cm = confusion_matrix(yte, ypred, labels=np.unique(y))
    acc = accuracy_score(yte, ypred)
    err_total = 1.0 - acc

    per_class_errors = {}
    for k in range(len(np.unique(y))):
        TP = cm[k, k]
        FN = cm[k, :].sum() - TP
        recall_k = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        per_class_errors[k] = 1.0 - recall_k

    cv = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=random_state)
    cv_acc = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    cv_err_mean = 1.0 - cv_acc.mean()
    cv_err_std = cv_acc.std(ddof=1)

    report = classification_report(yte, ypred, digits=3)
    return dict(clf=clf, Xte=Xte, yte=yte, ypred=ypred, cm=cm,
                acc=acc, err_total=err_total, per_class_errors=per_class_errors,
                cv_err_mean=cv_err_mean, cv_err_std=cv_err_std, report=report)

# --
# Визуализация: зоны решений
# --
def plot_regions(ax, X, y, clf, title):
    x_min, x_max = X[:,0].min() - 1.5, X[:,0].max() + 1.5
    y_min, y_max = X[:,1].min() - 1.5, X[:,1].max() + 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    XY = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(XY).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.16, levels=np.arange(-0.5, len(np.unique(y))+0.5, 1))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for k in np.unique(y):
        ax.scatter(X[y==k,0], X[y==k,1], s=16, label=f"Класс {k}", alpha=0.9,
                   edgecolor="white", linewidths=0.4, c=colors[k])
    ax.set_title(title)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

def show_confmat_side_by_side(cmA, cmB, titles=("A", "B")):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ConfusionMatrixDisplay(cmA, display_labels=[0,1,2,3]).plot(values_format="d", cmap="Blues", ax=axes[0], colorbar=False)
    axes[0].set_title(f"Confusion matrix — {titles[0]}")
    ConfusionMatrixDisplay(cmB, display_labels=[0,1,2,3]).plot(values_format="d", cmap="Blues", ax=axes[1], colorbar=False)
    axes[1].set_title(f"Confusion matrix — {titles[1]}")
    plt.tight_layout()
    plt.show()

# --
# Основной сценарий: два случая
# --
def main():
    # (A) Идеально разделимые
    XA, yA = generate_4_gaussians(n_per_class=300, dist=4.0, spread=0.25)
    resA = evaluate_svm(XA, yA, kernel='linear')

    # (B) Частично пересекающиеся
    XB, yB = generate_4_gaussians(n_per_class=300, dist=3.0, spread=5.2)
    resB = evaluate_svm(XB, yB, kernel='linear')

    #  Табличка в консоль 
    print(" Сводка по вероятности ошибок (SVM, kernel=linear) ")
    print("Случай A: Линейно разделимые")
    print(f"  Общая ошибка: {resA['err_total']:.4f} | CV ошибка: {resA['cv_err_mean']:.4f} (std={resA['cv_err_std']:.4f})")
    print("  Ошибка по классам:", ", ".join([f"{k}:{v:.4f}" for k,v in resA['per_class_errors'].items()]))
    print()
    print("Случай B: Пересечение классов")
    print(f"  Общая ошибка: {resB['err_total']:.4f} | CV ошибка: {resB['cv_err_mean']:.4f} (std={resB['cv_err_std']:.4f})")
    print("  Ошибка по классам:", ", ".join([f"{k}:{v:.4f}" for k,v in resB['per_class_errors'].items()]))

    #  Границы решений рядом 
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=False, sharey=False)
    plot_regions(axes[0], XA, yA, resA["clf"], "A) Линейно разделимые (kernel=linear)")
    plot_regions(axes[1], XB, yB, resB["clf"], "B) С пересечением (kernel=linear)")
    plt.tight_layout()
    plt.show()

    #  Матрицы ошибок рядом 
    show_confmat_side_by_side(resA["cm"], resB["cm"], titles=("A (разделимые)", "B (пересечение)"))

    # (необяз.) подробные отчёты по тесту — можно вставить в приложение отчёта
    print("\n Classification report — A (разделимые) ")
    print(resA["report"])
    print("\n Classification report — B (пересечение) ")
    print(resB["report"])

if __name__ == "__main__":
    main()
