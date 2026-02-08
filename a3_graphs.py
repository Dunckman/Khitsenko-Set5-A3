import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_graph1(csv_path, title_suffix, save_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df["step_percent"], df["exact"], "b-o", label="Точное $F_0^t$", linewidth=2)
    plt.plot(df["step_percent"], df["estimate"], "r--s", label="Оценка $N_t$ (HyperLogLog)", linewidth=2)
    plt.xlabel("Обработанная часть потока (%)", fontsize=13)
    plt.ylabel("Количество уникальных элементов", fontsize=13)
    plt.title(f"Сравнение $N_t$ и $F_0^t$ ({title_suffix})", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_graph2(csv_path, title_suffix, save_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df["step_percent"], df["mean_estimate"], "r-o", label="$E(N_t)$", linewidth=2)
    plt.plot(df["step_percent"], df["mean_exact"], "b--", label="Среднее точное $F_0^t$", linewidth=1.5, alpha=0.7)
    plt.fill_between(
        df["step_percent"], df["lower"], df["upper"],
        color="red", alpha=0.2, label="$E(N_t) \\pm \\sigma_{N_t}$"
    )
    plt.xlabel("Обработанная часть потока (%)", fontsize=13)
    plt.ylabel("Количество уникальных элементов", fontsize=13)
    plt.title(f"Статистики оценки HyperLogLog ({title_suffix})", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

for b in [4, 8, 10, 14]:
    prefix = f"B{b}"
    label = f"B={b}, M={2**b}"
    plot_graph1(f"data/{prefix}_graph1.csv", label, f"graphs/{prefix}_graph1.png")
    plot_graph2(f"data/{prefix}_graph2.csv", label, f"graphs/{prefix}_graph2.png")

plt.figure(figsize=(10, 6))
bs = [4, 8, 10, 14]
final_errors = []
final_stds = []
for b in bs:
    df = pd.read_csv(f"data/B{b}_analysis.csv")
    last = df.iloc[-1]
    final_errors.append(last["rel_error_pct"])
    final_stds.append(last["rel_std_pct"])

b_range = np.arange(4, 15)
theory1 = [1.042 / np.sqrt(2**b) * 100 for b in b_range]
theory2 = [1.3 / np.sqrt(2**b) * 100 for b in b_range]

plt.plot(b_range, theory1, "g--", label="$1.042/\\sqrt{2^B}$ (теория)", linewidth=2)
plt.plot(b_range, theory2, "m--", label="$1.3/\\sqrt{2^B}$ (теория)", linewidth=2)
plt.scatter(bs, final_stds, color="red", s=100, zorder=5, label="Практическое $\\sigma/F_0^t$")
plt.scatter(bs, final_errors, color="blue", s=100, zorder=5, marker="^", label="Практическая |ошибка E|")
plt.xlabel("B (бит индекса)", fontsize=13)
plt.ylabel("Относительная ошибка / отклонение (%)", fontsize=13)
plt.title("Зависимость точности HyperLogLog от параметра B", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(b_range)
plt.tight_layout()
plt.savefig("graphs/summary_B_comparison.png", dpi=150)
plt.close()

print("Все графики сохранены")