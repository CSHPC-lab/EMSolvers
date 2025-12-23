import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 設定
csv_dir = "csv"  # CSVファイルがあるディレクトリ
plot_dir = os.path.join(csv_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)
dx = 0.01  # グリッド間隔の基準値

# プロット対象列
columns = ["ExEM", "ExPM", "EyEM", "EyPM", "EzEM", "EzPM"]

# CSVファイル一覧
csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

# 各列ごとにlog-logプロット
for col in columns:
    plt.figure()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if col not in df.columns:
            continue
        prefix = os.path.splitext(os.path.basename(csv_file))[0].replace("ana-", "")
        y = df[col].astype(float).values
        n = np.arange(1, len(y) + 1)
        x = dx / n

        # 収束次数を計算（最初と最後の点を使用）
        x1, y1 = x[0], y[0]  # 最初の点
        x2, y2 = x[-1], y[-1]  # 最後の点
        order = (np.log(y2) - np.log(y1)) / (np.log(x2) - np.log(x1))

        plt.loglog(x, y, marker="o", label=f"{prefix} (order: {order:.2f})")
        print(f"  {prefix}: {order:.4f}")

    plt.xlabel("Δx (log scale)")
    plt.ylabel(col + " (log scale)")
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{col}_loglog.png"))
    plt.close()

print(f"✅ 両対数プロット画像を {plot_dir} に出力しました。")
