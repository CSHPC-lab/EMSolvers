import os
import glob
import pandas as pd
import re

def convert_csv_to_dat():
    csv_files = sorted(glob.glob("*.csv"))
    dat_files = []
    dt_map = {}

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if not {'u', 't'}.issubset(df.columns):
            print(f"Skipping {csv_file}: Missing required columns.")
            continue

        if len(df) < 2:
            print(f"Skipping {csv_file}: Not enough rows to calculate dt.")
            continue

        # DTを自動計算
        dt = df['t'].iloc[1] - df['t'].iloc[0]
        dt_map[csv_file] = dt

        output_lines = []
        for idx, row in df.iterrows():
            t = idx * dt  # 実時間ではなく再構成
            line = f"{t:.2f} {row['u']:.6E}"
            output_lines.append(line)

        base = os.path.splitext(csv_file)[0]
        dat_file = base + ".dat"
        with open(dat_file, 'w') as f:
            f.write('\n'.join(output_lines))

        dat_files.append(dat_file)
        print(f"Converted: {csv_file} -> {dat_file} (dt={dt:.6g})")

    return dat_files, dt_map

def update_input_file(filename, mt, dt, s1_name, s2_name):
    with open(filename, 'r') as f:
        content = f.read()

    content = re.sub(r'MT\s*=\s*[^,]+', f'MT={mt}', content)
    content = re.sub(r'DT\s*=\s*[^,]+', f'DT={dt:.6g}', content)
    content = re.sub(r"S1_NAME\s*=\s*'[^']+'", f"S1_NAME='{s1_name}'", content)
    content = re.sub(r"S2_NAME\s*=\s*'[^']+'", f"S2_NAME='{s2_name}'", content)

    with open(filename, 'w') as f:
        f.write(content)

    print(f"{filename} を更新しました。")

def main():
    dat_files, dt_map = convert_csv_to_dat()

    if len(dat_files) != 2:
        print("変換された .dat ファイルが2つではありません。処理を中止します。")
        return

    # 対応するCSVファイル名も取得
    csv_files = sorted(glob.glob("*.csv"))
    s1_csv = csv_files[0]
    s2_csv = csv_files[1]
    s1_dat = os.path.splitext(s1_csv)[0] + ".dat"
    s2_dat = os.path.splitext(s2_csv)[0] + ".dat"

    # 行数を取得（= MT）
    with open(s1_dat) as f:
        mt = sum(1 for _ in f)

    # DTをs1_csvから取得
    dt = dt_map[s1_csv]

    update_input_file('HF_TF-MISFIT_GOF', mt, dt, s1_dat, s2_dat)

if __name__ == "__main__":
    main()
