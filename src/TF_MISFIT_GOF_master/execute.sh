#!/bin/bash

# 作業ディレクトリ
TARGETS_DIR="./target"
WORK_DIR="."
OUTPUT_DIR="./output"

# targetsディレクトリ内の全ファイルをループ処理
for file in "$TARGETS_DIR"/*; do
    filename=$(basename "$file")
    echo "Processing $filename"

    # 一つ上のディレクトリに移動
    mv "$file" "$WORK_DIR"

    # main.pyとtf_misfits_gofを実行
    python main.py
    ./tf_misfits_gof

    # .csvファイルのうち最初の2つを取得
    csv_files=($(find "$WORK_DIR" -maxdepth 1 -type f -name "*.csv" | sort))
    if [ "${#csv_files[@]}" -lt 2 ]; then
        echo "Error: Less than two CSV files found."
        exit 1
    fi

    csv1=$(basename "${csv_files[0]}" .csv)
    csv2=$(basename "${csv_files[1]}" .csv)

    # MISFIT-GOF.DAT をリネーム
    old="$WORK_DIR/MISFIT-GOF.DAT"
    new="$OUTPUT_DIR/${csv1}-${csv2}.dat"

    if [ -e "$old" ]; then
        mv "$old" "$new"
        echo "Renamed: $old -> $new"
    else
        echo "Error: $old does not exist."
    fi

    # *.DAT ファイルを削除
    rm -f "$WORK_DIR"/*.DAT
    rm -f "$WORK_DIR"/*.dat

    # 処理済みファイルをtargetsに戻す
    mv "$WORK_DIR/$filename" "$TARGETS_DIR"
done