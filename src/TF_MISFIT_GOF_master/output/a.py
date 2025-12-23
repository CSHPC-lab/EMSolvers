import glob
import os
import csv
from collections import defaultdict

# 1. *_1.dat ～ *_8.dat をすべて取得
all_files = glob.glob('./*_[1-8]*')

# 2. 接頭辞ごとにグループ化
groups = defaultdict(list)
for filepath in all_files:
    filename = os.path.basename(filepath)
    prefix = filename.split('_')[0]  # "abc_1.dat" → "abc"
    groups[prefix].append(filepath)

# ヘッダー
headers = ["ExEM", "ExPM", "EyEM", "EyPM", "EzEM", "EzPM"]

# 3. 各 prefix グループごとに処理
for prefix, filepaths in groups.items():
    # ファイル番号順にソート（~_1_~.dat 〜 ~_8_~.dat の順にする）
    sorted_paths = sorted(filepaths, key=lambda x: int(x.split('_')[1]))

    rows = []
    for path in sorted_paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 7:
                raise ValueError(f"{path} は7行未満です")
            row = []
            for line in lines[4:7]:  # 5〜7行目（0-indexed）
                row.extend(line.strip().split())  # 2つずつ
            rows.append(row)  # 各行：6要素

    # 出力先ファイル名を作成
    output_csv = f"csv/{os.path.basename(prefix)}.csv"
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"✅ {output_csv} を出力しました。")
