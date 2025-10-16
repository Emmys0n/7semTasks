import os
import re
import warnings
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# КОНФИГ
INPUT_PATH = "data.csv"
N_ROWS = 1000

FIG_DIR = "figs"
OUT_CLEAN = "data_cleaned.csv"

# Правила по пропускам
COL_DROP_MISSING_RATIO = 0.60  # удалить столбцы, где >60% NaN
ROW_DROP_MISSING_RATIO = 0.50  # удалить строки, где >50% NaN

# Какие числовые поля попробовать для boxplot (если есть)
BOXPLOT_PREFS: List[str] = [
    "price_doc", "full_sq", "life_sq", "kitch_sq",
    "num_room", "floor", "max_floor", "build_year"
]
# ============================================

os.makedirs(FIG_DIR, exist_ok=True)
warnings.filterwarnings("ignore")  # не шумим в консоли


# Надежное чтение CSV
def robust_read_csv(path, nrows=None):
    encodings = ["utf-8", "utf-8-sig", "cp1251", "windows-1251", "latin1"]
    seps = [None, ",", ";", "\t"]
    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(path, sep=sep, encoding=enc, engine="python", nrows=nrows)
            except Exception:
                continue
    return pd.read_csv(path, sep=None, encoding="latin1", engine="python",
                       on_bad_lines="skip", nrows=nrows)


def main():
    # 1) Загрузка
    df = robust_read_csv(INPUT_PATH, nrows=N_ROWS)
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]

    # 2) Анализ пропусков
    miss_ratio = df.isna().mean().sort_values(ascending=False)

    # (1) Теплокарта пропусков
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Тепловая карта пропусков")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "1_heatmap_missing.png"), dpi=150)
    plt.close()

    # (2) Бар по пропускам
    plt.figure(figsize=(max(10, 0.15 * len(df.columns) + 6), 6))
    miss_ratio.plot(kind="bar")
    plt.title("Проценты пропусков по столбцам")
    plt.ylabel("%")
    plt.xticks(rotation=90, fontsize=6)
    plt.subplots_adjust(bottom=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "2_missing_bar.png"), dpi=150)
    plt.close()

    # 3) Очистка пропусков (минимум)
    # Удаляем "тяжелые" по пропускам столбцы и строки
    drop_cols = miss_ratio[miss_ratio > COL_DROP_MISSING_RATIO].index.tolist()
    df = df.drop(columns=drop_cols) if drop_cols else df
    df = df.loc[df.isna().mean(axis=1) <= ROW_DROP_MISSING_RATIO].copy()

    # Импутация по типу: числа -> медиана, строки -> мода
    for col in df.columns:
        if df[col].isna().any():
            if np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])
                else:
                    df[col] = df[col].fillna("")

    # 4) Boxplot по числовым
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Выбираем 2–3 понятных числовых признака: сперва из предпочтительных, иначе первые по дисперсии
    chosen = [c for c in BOXPLOT_PREFS if c in num_cols][:3]
    if not chosen:
        # берём топ-3 по дисперсии, чтобы график был информативным
        var = df[num_cols].var().sort_values(ascending=False)
        chosen = list(var.head(3).index)

    if chosen:
        # (3) Один общий boxplot для выбранных 2–3 признаков
        plt.figure(figsize=(8, 4 + 0.6 * len(chosen)))
        sns.boxplot(data=df[chosen], orient="h")
        plt.title("Boxplot для выбранных числовых признаков")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "3_boxplot_outliers.png"), dpi=150)
        plt.close()

    # 5) Удаление дубликатов
    before = len(df)
    df = df.drop_duplicates().copy()
    removed_dups = before - len(df)

    # 6) Сохранение результата
    df.to_csv(OUT_CLEAN, index=False, encoding="utf-8")

    # 7) КОНСОЛЬНЫЙ ОТЧЁТ
    lines = []
    lines.append("=== ОТЧЕТ ПО ОЧИСТКЕ ДАННЫХ (краткий) ===")
    lines.append(f"Исходно: {N_ROWS} строк × {len(miss_ratio)} столбцов")
    lines.append(f"Удалено столбцов с >{int(COL_DROP_MISSING_RATIO*100)}% NaN: {len(drop_cols)}")
    if drop_cols:
        lines.append("  " + ", ".join(drop_cols[:10]) + ("..." if len(drop_cols) > 10 else ""))
    lines.append(f"Удалено дубликатов строк: {removed_dups}")
    lines.append(f"Итоговый размер: {df.shape[0]} строк × {df.shape[1]} столбцов")
    lines.append("")
    lines.append("Топ-15 по доле пропусков (до очистки):")
    top15 = (miss_ratio.head(15) * 100).round(2)
    lines.append(str(top15))
    lines.append("")
    if chosen:
        lines.append("Boxplot построен для признаков: " + ", ".join(chosen))
        desc = df[chosen].describe().round(3)
        lines.append("\nОписательная статистика выбранных числовых признаков:")
        lines.append(str(desc))
    else:
        lines.append("Числовых столбцов не найдено — boxplot не строился.")
    lines.append("")
    lines.append(f"Сохранено: {OUT_CLEAN}")
    lines.append(f"Картинки: figs/1_heatmap_missing.png, figs/2_missing_bar.png, figs/3_boxplot_outliers.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
