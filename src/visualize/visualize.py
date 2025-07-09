import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_multiple_csv_files(directory='inputs'):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    dataframes = []
    file_names = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if 'time' not in df.columns:
                print(f"Skipping {file}: 'time' column not found.")
                continue
            dataframes.append(df)
            file_names.append(os.path.basename(file))
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not dataframes:
        print("No valid CSV files were loaded.")
        return

    field_cols = ['Ex', 'Ey', 'Ez']
    existing_fields = [col for col in field_cols if any(col in df.columns for df in dataframes)]

    fig, axes = plt.subplots(len(existing_fields), 1, figsize=(12, 3 * len(existing_fields)), sharex=True)
    if len(existing_fields) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, len(dataframes)))

    for i, field in enumerate(existing_fields):
        ax = axes[i]
        for j, (df, name) in enumerate(zip(dataframes, file_names)):
            if field in df.columns:
                ax.plot(df['time'], df[field], label=name, color=colors[j], linewidth=1.5)
        ax.set_ylabel(f"{field} [V/m]")
        ax.set_title(f"{field} Time Series")
        ax.grid(True)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.legend(fontsize='small', loc='upper right')

    axes[-1].set_xlabel('Time [s]')
    # fig.suptitle('Comparison of Electric Field Components', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('comparison_time_series_xyz.png')
    print("Saved: comparison_time_series_xyz.png")
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
    if 'time' in df.columns and 'Ex' in df.columns:
        time_col = 'time'
        ex_col = 'Ex'
    elif len(df.columns) == 2:
        df = df.rename(columns={df.columns[0]: 'time', df.columns[1]: 'Ex'})
        time_col = 'time'
        ex_col = 'Ex'
    else:
        print(f"Expected columns 'time' and 'Ex' not found.")
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
    plt.specgram(df[ex_col].values, Fs=fs, cmap='viridis')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram of Electric Field (Ex)')
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
    if 'time' in df.columns and 'Ex' in df.columns:
        time_col = 'time'
        ex_col = 'Ex'
    elif len(df.columns) == 2:
        df = df.rename(columns={df.columns[0]: 'time', df.columns[1]: 'Ex'})
        time_col = 'time'
        ex_col = 'Ex'
    else:
        print(f"Expected columns 'time' and 'Ex' not found.")
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
    signal = df[ex_col].values
    
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
    plt.title('Frequency Spectrum of Electric Field (Ex)')
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
