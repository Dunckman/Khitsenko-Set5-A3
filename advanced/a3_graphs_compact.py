import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_compact_graph1(csv_path, title_suffix, save_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df["step_percent"], df["exact"], "b-o", label="Точное $F_0^t$", linewidth=2)
    plt.plot(df["step_percent"], df["estimate_standard"], "r--s",
             label="Стандартный HLL", linewidth=2)
    plt.plot(df["step_percent"], df["estimate_compact"], "g:^",
             label="Компактный HLL (5 бит/рег)", linewidth=2)
    plt.xlabel("Обработанная часть потока (%)", fontsize=13)
    plt.ylabel("Количество уникальных элементов", fontsize=13)
    plt.title(f"Сравнение стандартного и компактного HLL ({title_suffix})", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_compact_graph2(csv_path, title_suffix, save_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df["step_percent"], df["mean_estimate"], "g-o",
             label="$E(N_t)$ компактный", linewidth=2)
    plt.plot(df["step_percent"], df["mean_exact"], "b--",
             label="Среднее точное $F_0^t$", linewidth=1.5, alpha=0.7)
    plt.fill_between(
        df["step_percent"], df["lower"], df["upper"],
        color="green", alpha=0.2, label="$E(N_t) \\pm \\sigma_{N_t}$"
    )
    plt.xlabel("Обработанная часть потока (%)", fontsize=13)
    plt.ylabel("Количество уникальных элементов", fontsize=13)
    plt.title(f"Статистики компактного HyperLogLog ({title_suffix})", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# Графики для каждого B
for b in [4, 8, 10, 14]:
    prefix = f"B{b}"
    label = f"B={b}, M={2**b}"
    plot_compact_graph1(f"data/{prefix}_compact_graph1.csv", label,
                        f"graphs/{prefix}_compact_graph1.png")
    plot_compact_graph2(f"data/{prefix}_compact_graph2.csv", label,
                        f"graphs/{prefix}_compact_graph2.png")

bs = [4, 8, 10, 14]
mem_std = [2**b * 1 for b in bs]
mem_cmp = [(2**b * 5 + 7) // 8 for b in bs]

fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(bs))
width = 0.35
bars1 = ax1.bar(x - width/2, mem_std, width, label="Стандартный (8 бит/рег)",
                color="salmon", edgecolor="black")
bars2 = ax1.bar(x + width/2, mem_cmp, width, label="Компактный (5 бит/рег)",
                color="lightgreen", edgecolor="black")

ax1.set_xlabel("B (бит индекса)", fontsize=13)
ax1.set_ylabel("Память (байт)", fontsize=13)
ax1.set_title("Сравнение потребления памяти", fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels([f"B={b}" for b in bs])
ax1.legend(fontsize=12)
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3, axis="y")

for bar in bars1:
    ax1.annotate(f"{int(bar.get_height())}",
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 ha="center", va="bottom", fontsize=10)
for bar in bars2:
    ax1.annotate(f"{int(bar.get_height())}",
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("graphs/memory_comparison.png", dpi=150)
plt.close()

print("Все графики этапа 4 сохранены")