import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_time_series(filename):
    """
    Plot time series data from a CSV file with 't' and 'Ez' columns.
    """
    # CSVファイルの読み込み
    try:
        df = pd.read_csv(filename)
        print(f"Loaded data from {filename}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # ヘッダー名のチェックと統一
    # tとEzの列名を確認し、必要に応じてリネーム
    if 't' in df.columns and 'Ez' in df.columns:
        time_col = 't'
        ez_col = 'Ez'
    elif 'time' in df.columns.str.lower() and 'ez' in df.columns.str.lower():
        time_col = df.columns[df.columns.str.lower() == 'time'][0]
        ez_col = df.columns[df.columns.str.lower() == 'ez'][0]
        df = df.rename(columns={time_col: 't', ez_col: 'Ez'})
        time_col = 't'
        ez_col = 'Ez'
    else:
        # 列が2つしかない場合は最初の列を時間、2番目を電界と仮定
        if len(df.columns) == 2:
            df = df.rename(columns={df.columns[0]: 't', df.columns[1]: 'Ez'})
            time_col = 't'
            ez_col = 'Ez'
            print(f"Assuming columns are time and Ez: {df.columns.tolist()}")
        else:
            print(f"Expected columns 't' and 'Ez' not found.")
            print(f"Available columns: {df.columns.tolist()}")
            return

    # 電界(Ez)の時系列プロット
    plt.figure(figsize=(12, 8))
    plt.plot(df[time_col], df[ez_col], label='Ez', color='blue', linewidth=1.5)
    
    plt.xlabel('Time [s]')
    plt.ylabel('Electric Field Ez [V/m]')
    plt.title('Electric Field (Ez) Time Series')
    plt.legend()
    plt.grid(True)
    
    # グラフの保存
    output_filename = 'electric_field_time_series.png'
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")
    
    # 科学的表記のx軸ラベルを設定
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    
    # 視覚的な改善を適用
    plt.tight_layout()
    
    # インタラクティブモードでの表示 (オプション)
    plt.show()
    plt.close()

def plot_spectrogram(filename):
    """
    Create a spectrogram from the time series data.
    """
    # CSVファイルの読み込み
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # ヘッダー名のチェックと統一
    if 't' in df.columns and 'Ez' in df.columns:
        time_col = 't'
        ez_col = 'Ez'
    elif len(df.columns) == 2:
        df = df.rename(columns={df.columns[0]: 't', df.columns[1]: 'Ez'})
        time_col = 't'
        ez_col = 'Ez'
    else:
        print(f"Expected columns 't' and 'Ez' not found.")
        return

    # サンプリング周波数の計算
    time_values = df[time_col].values
    if len(time_values) > 1:
        # 平均的なサンプリング間隔を計算
        avg_dt = np.mean(np.diff(time_values))
        fs = 1.0 / avg_dt
    else:
        print("Not enough time points for spectrogram")
        return
    
    # スペクトログラムの作成
    plt.figure(figsize=(12, 8))
    plt.specgram(df[ez_col].values, Fs=fs, cmap='viridis')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram of Electric Field (Ez)')
    plt.colorbar(label='Intensity [dB]')
    
    # グラフの保存
    output_filename = 'electric_field_spectrogram.png'
    plt.savefig(output_filename)
    print(f"Saved spectrogram to {output_filename}")
    
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_frequency_spectrum(filename):
    """
    Plot the frequency spectrum of the time series data.
    """
    # CSVファイルの読み込み
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # ヘッダー名のチェックと統一
    if 't' in df.columns and 'Ez' in df.columns:
        time_col = 't'
        ez_col = 'Ez'
    elif len(df.columns) == 2:
        df = df.rename(columns={df.columns[0]: 't', df.columns[1]: 'Ez'})
        time_col = 't'
        ez_col = 'Ez'
    else:
        print(f"Expected columns 't' and 'Ez' not found.")
        return

    # サンプリング周波数の計算
    time_values = df[time_col].values
    if len(time_values) > 1:
        # 平均的なサンプリング間隔を計算
        avg_dt = np.mean(np.diff(time_values))
        fs = 1.0 / avg_dt
    else:
        print("Not enough time points for frequency analysis")
        return
    
    # 信号データ
    signal = df[ez_col].values
    
    # FFTの実行
    n = len(signal)
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(n, avg_dt)
    
    # 正の周波数のみプロット
    positive_freq_idx = xf > 0
    
    plt.figure(figsize=(12, 8))
    plt.plot(xf[positive_freq_idx], 2.0/n * np.abs(yf[positive_freq_idx]), linewidth=1.5)
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum of Electric Field (Ez)')
    plt.grid(True)
    
    # 対数スケールのオプション
    # plt.xscale('log')
    # plt.yscale('log')
    
    # グラフの保存
    output_filename = 'electric_field_spectrum.png'
    plt.savefig(output_filename)
    print(f"Saved frequency spectrum to {output_filename}")
    
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_multiple_csv_files(directory='inputs'):
    """
    Plot multiple CSV files on the same graph for comparison.
    """
    # CSVファイルを検索
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {directory}: {[os.path.basename(f) for f in csv_files]}")
    
    # 全ファイルのデータをロード
    dataframes = []
    file_names = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # ヘッダー名のチェックと統一
            if 't' in df.columns and 'Ez' in df.columns:
                pass
            elif len(df.columns) == 2:
                df = df.rename(columns={df.columns[0]: 't', df.columns[1]: 'Ez'})
            else:
                print(f"Skipping {file}: Expected columns 't' and 'Ez' not found.")
                continue
            
            dataframes.append(df)
            file_names.append(os.path.basename(file))
            print(f"Loaded {os.path.basename(file)}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not dataframes:
        print("No valid CSV files were loaded.")
        return
    
    # 時系列の比較プロット
    plt.figure(figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(dataframes)))
    
    for i, (df, name) in enumerate(zip(dataframes, file_names)):
        plt.plot(df['t'], df['Ez'], label=name, color=colors[i], linewidth=1.5)
    
    plt.xlabel('Time [s]')
    plt.ylabel('Electric Field Ez [V/m]')
    plt.title('Comparison of Electric Field (Ez) Time Series')
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.tight_layout()
    
    # グラフの保存
    output_filename = 'comparison_time_series.png'
    plt.savefig(output_filename)
    print(f"Saved comparison plot to {output_filename}")
    plt.close()
    
    # 周波数スペクトルの比較
    plt.figure(figsize=(14, 10))
    
    for i, (df, name) in enumerate(zip(dataframes, file_names)):
        # サンプリング周波数の計算
        time_values = df['t'].values
        if len(time_values) > 1:
            avg_dt = np.mean(np.diff(time_values))
            fs = 1.0 / avg_dt
            
            # 信号データ
            signal = df['Ez'].values
            
            # FFTの実行
            n = len(signal)
            yf = np.fft.fft(signal)
            xf = np.fft.fftfreq(n, avg_dt)
            
            # 正の周波数のみプロット
            positive_freq_idx = xf > 0
            plt.plot(xf[positive_freq_idx], 2.0/n * np.abs(yf[positive_freq_idx]), 
                    label=name, color=colors[i], linewidth=1.5)
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Comparison of Frequency Spectra')
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.tight_layout()
    
    # グラフの保存
    output_filename = 'comparison_spectrum.png'
    plt.savefig(output_filename)
    print(f"Saved frequency spectrum comparison to {output_filename}")
    plt.close()

def process_csv_file(filename):
    """
    Process a single CSV file, generating all visualizations.
    """
    print(f"Processing file: {filename}")
    plot_time_series(filename)
    plot_spectrogram(filename)
    plot_frequency_spectrum(filename)
    print(f"Completed processing {filename}")

if __name__ == "__main__":
    # パラメータの処理
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # 比較モード: inputsディレクトリの全てのCSVファイルを比較
        inputs_dir = "inputs"
        if len(sys.argv) > 2:
            inputs_dir = sys.argv[2]
        plot_multiple_csv_files(inputs_dir)
    else:
        # 単一ファイルモード
        # デフォルトのファイルパス
        default_file = "../fdtd/observation.csv"
        
        # ファイルが存在するか確認
        if os.path.exists(default_file):
            process_csv_file(default_file)
        else:
            print(f"Default file {default_file} not found.")
            
            # 代替パスを探す
            alternative_paths = [
                "observation.csv",
                "src/fdtd/observation.csv",
                "inputs/observation.csv"
            ]
            
            file_found = False
            for path in alternative_paths:
                if os.path.exists(path):
                    print(f"Found alternative file at {path}")
                    process_csv_file(path)
                    file_found = True
                    break
            
            if not file_found:
                print("No observation.csv file found. Trying to compare files in inputs directory instead.")
                plot_multiple_csv_files()
    
    print("\n可視化が完了しました。以下のファイルが生成されました：")
    print("- electric_field_time_series.png (時間領域の電界波形)")
    print("- electric_field_spectrogram.png (時間-周波数解析)")
    print("- electric_field_spectrum.png (周波数スペクトル)")
    print("- comparison_time_series.png (複数ファイルの時系列比較) - 比較モード時のみ")
    print("- comparison_spectrum.png (複数ファイルの周波数スペクトル比較) - 比較モード時のみ")
    print("\n比較モードで実行するには: python visualize.py --compare [inputs_directory]")
