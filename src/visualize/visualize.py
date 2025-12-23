import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys


def plot_multiple_csv_files(directory="inputs"):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    csv_files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")
    ]
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    dataframes = []
    file_names = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if "time" not in df.columns:
                print(f"Skipping {file}: 'time' column not found.")
                continue
            dataframes.append(df)
            file_names.append(os.path.basename(file))
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if not dataframes:
        print("No valid CSV files were loaded.")
        return

    field_cols = ["Ex", "Ey", "Ez"]
    existing_fields = [
        col for col in field_cols if any(col in df.columns for df in dataframes)
    ]

    fig, axes = plt.subplots(
        len(existing_fields), 1, figsize=(12, 3 * len(existing_fields)), sharex=True
    )
    if len(existing_fields) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, len(dataframes)))

    for i, field in enumerate(existing_fields):
        ax = axes[i]
        for j, (df, name) in enumerate(zip(dataframes, file_names)):
            if field in df.columns:
                ax.plot(
                    df["time"], df[field], label=name, color=colors[j], linewidth=1.5
                )
        ax.set_ylabel(f"{field} [V/m]")
        ax.set_title(f"{field} Time Series")
        ax.grid(True)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.legend(fontsize="small", loc="center left", bbox_to_anchor=(1, 0.5))
        # ax.set_xlim(2.0e-9, 2.3e-9)
        # ax.set_ylim(6.0e9, 8.3e9)
        # ax.set_xlim(5.5e-9, 5.8e-9)
        # ax.set_ylim(1.0e9, 2.1e9)

    axes[-1].set_xlabel("Time [s]")
    # fig.suptitle('Comparison of Electric Field Components', fontsize=14)
    fig.subplots_adjust(right=0.85)  # Adjust layout to make room for legend
    plt.savefig("outputs/comparison_time_series.png")
    print("Saved: comparison_time_series.png")
    plt.show()
    plt.close()


def plot_multiple_frequency_spectrum(directory="inputs"):
    """
    Plot the frequency spectrum for Ex/Ey/Ez in all CSV files, overlaid in one figure.
    Save both full and 0–10GHz limited versions.
    """
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    field_cols = ["Ex", "Ey", "Ez"]
    for col in field_cols:
        fig_full = plt.figure(figsize=(12, 8))
        ax_full = fig_full.add_subplot(111)

        fig_limited = plt.figure(figsize=(12, 8))
        ax_limited = fig_limited.add_subplot(111)

        colors = plt.cm.viridis(np.linspace(0, 1, len(csv_files)))

        for idx, file in enumerate(csv_files):
            try:
                df = pd.read_csv(file)
                if col not in df.columns or "time" not in df.columns:
                    print(f"Skipping {file}: required columns not found.")
                    continue

                time_values = df["time"].values
                avg_dt = np.mean(np.diff(time_values))

                signal = df[col].values
                n = len(signal)
                yf = np.fft.fft(signal)
                xf = np.fft.fftfreq(n, avg_dt)

                # フル帯域
                pos_idx = xf > 0
                ax_full.plot(
                    xf[pos_idx],
                    2.0 / n * np.abs(yf[pos_idx]),
                    label=os.path.basename(file),
                    linewidth=1.5,
                    color=colors[idx],
                )

                # 5GHzまでに絞った帯域
                lim_idx = (xf > 0) & (xf < 5e9)
                ax_limited.plot(
                    xf[lim_idx],
                    2.0 / n * np.abs(yf[lim_idx]),
                    label=os.path.basename(file),
                    linewidth=1.5,
                    color=colors[idx],
                )

            except Exception as e:
                print(f"Error processing {file}: {e}")

        # フル帯域プロット設定
        ax_full.set_xlabel("Frequency [Hz]")
        ax_full.set_ylabel("Amplitude")
        ax_full.set_title(f"Frequency Spectrum of Electric Field ({col})")
        ax_full.grid(True)
        ax_full.legend(fontsize="small", loc="center left", bbox_to_anchor=(1, 0.5))
        ax_full.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        fig_full.tight_layout(rect=[0, 0, 0.85, 1])
        os.makedirs("outputs", exist_ok=True)
        fig_full.savefig(f"outputs/frequency_spectrum_{col}.png")
        print(f"Saved: outputs/frequency_spectrum_{col}.png")

        # 限定帯域プロット設定
        ax_limited.set_xlabel("Frequency [Hz]")
        ax_limited.set_ylabel("Amplitude")
        ax_limited.set_title(f"Frequency Spectrum 0–5 GHz of Electric Field ({col})")
        ax_limited.grid(True)
        ax_limited.legend(fontsize="small", loc="center left", bbox_to_anchor=(1, 0.5))
        ax_limited.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        fig_limited.tight_layout(rect=[0, 0, 0.85, 1])
        fig_limited.savefig(f"outputs/frequency_spectrum_0-5GHz_{col}.png")
        print(f"Saved: outputs/frequency_spectrum_0-5GHz_{col}.png")

        plt.close(fig_full)
        plt.close(fig_limited)


if __name__ == "__main__":
    inputs_dir = "inputs"
    plot_multiple_csv_files(inputs_dir)
    plot_multiple_frequency_spectrum(inputs_dir)

    print("\n可視化が完了しました。以下のファイルが生成されました：")
    print("- electric_field_time_series.png (時間領域の電界波形)")
    print("- electric_field_spectrogram.png (時間-周波数解析)")
    print("- electric_field_spectrum.png (周波数スペクトル)")
    print("- comparison_time_series.png (複数ファイルの時系列比較) - 比較モード時のみ")
    print(
        "- comparison_spectrum.png (複数ファイルの周波数スペクトル比較) - 比較モード時のみ"
    )
    print(
        "\n比較モードで実行するには: python visualize.py --compare [inputs_directory]"
    )
