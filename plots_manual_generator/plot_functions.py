import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import time
import pandas as pd
from pathlib import Path


def err_vs_freq(df_results, fy, output_dir="./images"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_types = list(fy.keys())
    markers_by_bits = {
        "8": "o",
        "12": "^",
        "16": "s",
        "24": "D",
        "32": "*",
        "Ideal (NoQuant)": "X",
    }
    
    for bin_val in bin_types:
        safe_bin_name = str(bin_val).replace("(", "_").replace(")", "").replace(".", "_")

        for stale_file in output_dir.glob(
            f"complex_error_freq_{safe_bin_name}_bits_*.png"
        ):
            stale_file.unlink()
        stale_bin_plot = output_dir / f"complex_error_freq_{safe_bin_name}.png"
        if stale_bin_plot.exists():
            stale_bin_plot.unlink()

    for stale_file in output_dir.glob("magnitude_error_freq_*.png"):
        stale_file.unlink()

    df_magnitude = df_results[df_results["BinType"].isin(bin_types)].copy()
    df_magnitude["Bits"] = df_magnitude["Bits"].astype(str)
    if not df_magnitude.empty:
        architectures = df_magnitude["Architecture"].unique()
        bit_levels = sorted(
            df_magnitude["Bits"].unique(),
            key=lambda bits: (not bits.isdigit(), int(bits) if bits.isdigit() else bits),
        )
        bin_levels = [bin_type for bin_type in bin_types if bin_type in df_magnitude["BinType"].values]
        palette = sns.color_palette(n_colors=len(architectures))
        architecture_colors = dict(zip(architectures, palette))
        markers = {bits: markers_by_bits.get(bits, "X") for bits in bit_levels}
        bin_linestyles = {
            bin_type: ("--" if "off_bin" in str(bin_type) else "-")
            for bin_type in bin_levels
        }

        grouped = (
            df_magnitude.groupby(
                ["Architecture", "BinType", "Bits", "Frequency"],
                as_index=False,
                observed=True,
            )["Magnitude error"]
            .mean()
        )

        fig, ax = plt.subplots(figsize=(11, 6))
        for (architecture, bin_type, bits), series in grouped.groupby(
            ["Architecture", "BinType", "Bits"], sort=False, observed=True
        ):
            series = series.sort_values("Frequency")
            ax.plot(
                series["Frequency"],
                series["Magnitude error"],
                color=architecture_colors[architecture],
                linestyle=bin_linestyles[bin_type],
                marker=markers[bits],
                markersize=4,
                markevery=max(1, len(series) // 15),
            )

        legend_handles = [
            Line2D([], [], color=architecture_colors[architecture], label=architecture)
            for architecture in architectures
        ]
        legend_handles.extend(
            Line2D(
                [], [], color="black", marker=markers[bits], linestyle="None",
                label=f"Bits: {bits}",
            )
            for bits in bit_levels
        )
        legend_handles.extend(
            Line2D(
                [], [], color="black", linestyle=bin_linestyles[bin_type],
                label=f"{'Off-bin' if 'off_bin' in str(bin_type) else 'On-bin'}: {bin_type}",
            )
            for bin_type in bin_levels
        )

        ax.set_yscale("log")
        ax.minorticks_on()
        ax.set_title("Magnitude error vs frequency")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Spectral Magnitude Error")
        ax.grid(True, which="major", ls="-", alpha=0.6)
        ax.grid(True, which="minor", ls=":", alpha=0.3)
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(
            output_dir / "magnitude_error_freq_on_off_bin.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    df_complex = df_results[df_results["BinType"].isin(bin_types)].copy()
    df_complex["Bits"] = df_complex["Bits"].astype(str)
    if not df_complex.empty:
        architectures = df_complex["Architecture"].unique()
        bit_levels = sorted(
            df_complex["Bits"].unique(),
            key=lambda bits: (not bits.isdigit(), int(bits) if bits.isdigit() else bits),
        )
        bin_levels = [bin_type for bin_type in bin_types if bin_type in df_complex["BinType"].values]
        palette = sns.color_palette(n_colors=len(architectures))
        architecture_colors = dict(zip(architectures, palette))
        markers = {bits: markers_by_bits.get(bits, "X") for bits in bit_levels}
        bin_linestyles = {
            bin_type: ("--" if "off_bin" in str(bin_type) else "-")
            for bin_type in bin_levels
        }

        grouped = (
            df_complex.groupby(
                ["Architecture", "BinType", "Bits", "Frequency"],
                as_index=False,
                observed=True,
            )["Complex error"]
            .mean()
        )

        fig, ax = plt.subplots(figsize=(11, 6))
        for (architecture, bin_type, bits), series in grouped.groupby(
            ["Architecture", "BinType", "Bits"], sort=False, observed=True
        ):
            series = series.sort_values("Frequency")
            ax.plot(
                series["Frequency"],
                series["Complex error"],
                color=architecture_colors[architecture],
                linestyle=bin_linestyles[bin_type],
                marker=markers[bits],
                markersize=4,
                markevery=max(1, len(series) // 15),
            )

        legend_handles = [
            Line2D([], [], color=architecture_colors[architecture], label=architecture)
            for architecture in architectures
        ]
        legend_handles.extend(
            Line2D([], [], color="black", marker=markers[bits], linestyle="None",
                   label=f"Bits: {bits}")
            for bits in bit_levels
        )
        legend_handles.extend(
            Line2D([], [], color="black", linestyle=bin_linestyles[bin_type],
                   label=f"{'Off-bin' if 'off_bin' in str(bin_type) else 'On-bin'}: {bin_type}")
            for bin_type in bin_levels
        )

        ax.set_yscale("log")
        ax.minorticks_on()
        ax.set_title("Complex global error vs frequency")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Spectral Complex Error")
        ax.grid(True, which="major", ls="-", alpha=0.6)
        ax.grid(True, which="minor", ls=":", alpha=0.3)
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(
            output_dir / "complex_error_freq_on_off_bin.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def bit_err(df_results, fy, output_dir="./images"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob("magnitude_error_vs_bits_number_*.png"):
        stale_file.unlink()
    for stale_file in output_dir.glob("complex_error_vs_bits_number_*.png"):
        stale_file.unlink()

    df_quant_all = df_results[df_results["Bits"] != "Ideal (NoQuant)"].copy()
    if not df_quant_all.empty:
        df_quant_all["Bits_num"] = pd.to_numeric(df_quant_all["Bits"])
        bit_levels = sorted(df_quant_all["Bits_num"].unique(), reverse=True)
        bin_levels = list(df_quant_all["BinType"].unique())
        bin_labels = {
            bin_type: ("Off-bin" if "off_bin" in str(bin_type) else "On-bin")
            for bin_type in bin_levels
        }
        df_quant_all["Bin label"] = df_quant_all["BinType"].map(bin_labels)
        plot_order = list(dict.fromkeys(bin_labels.values()))
        bin_dashes = {
            label: (2, 2) if label == "Off-bin" else ""
            for label in plot_order
        }

        plt.figure(figsize=(9, 5.5))
        ax = sns.lineplot(
            data=df_quant_all,
            x="Bits_num",
            y="Magnitude error",
            hue="Architecture",
            style="Bin label",
            style_order=plot_order,
            dashes=bin_dashes,
            estimator="median",
            errorbar=("pi", 50),
            marker="o",
        )
        plt.yscale("log")
        plt.minorticks_on()
        plt.gca().invert_xaxis()
        plt.xticks(bit_levels, [str(int(bits)) for bits in bit_levels])
        plt.title("Magnitude Error vs Bit Variation")
        plt.xlabel("Number of Bits")
        plt.ylabel("Median Magnitude Error")
        plt.grid(True, which="major", ls="-", alpha=0.6)
        plt.grid(True, which="minor", ls=":", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            output_dir / "magnitude_error_vs_bits_number_on_off_bin.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(9, 5.5))
        ax = sns.lineplot(
            data=df_quant_all,
            x="Bits_num",
            y="Complex error",
            hue="Architecture",
            style="Bin label",
            style_order=plot_order,
            dashes=bin_dashes,
            estimator="median",
            errorbar=("pi", 50),
            marker="o",
        )
        plt.yscale("log")
        plt.minorticks_on()
        plt.gca().invert_xaxis()
        plt.xticks(bit_levels, [str(int(bits)) for bits in bit_levels])
        plt.title("Complex Error vs Bit Variation")
        plt.xlabel("Number of Bits")
        plt.ylabel("Median Complex Error")
        plt.grid(True, which="major", ls="-", alpha=0.6)
        plt.grid(True, which="minor", ls=":", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            output_dir / "complex_error_vs_bits_number_on_off_bin.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def heat_map(df_results, fy, output_dir="./images"):
    """
    Heat map between amplitude, frequency and error, including r_ideal and r_quant in titles.
    """
    import matplotlib.ticker as ticker  # Assicurati di importarlo se non è già presente

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    architectures = df_results['Architecture'].unique()
    bin_types = list(fy.keys())
    error_types = ['Magnitude error', 'Complex error']
    bit_levels = df_results['Bits'].unique()

    for bin_val in bin_types:
        safe_bin_name = str(bin_val).replace("(", "_").replace(")", "").replace(".", "_")
        df_bin = df_results[df_results['BinType'] == bin_val]
        
        for bits_val in bit_levels:
            df_subset = df_bin[df_bin['Bits'] == bits_val]
            if df_subset.empty:
                continue
                
            bits_str_safe = str(bits_val).replace(" ", "_").replace("(", "").replace(")", "")
            
            fig, axes = plt.subplots(len(error_types), len(architectures), 
                                     figsize=(len(architectures) * 4.5, 4.5 * len(error_types)), 
                                     sharex=True, sharey=False)

            if len(architectures) == 1:
                axes = axes[:, np.newaxis]

            for r, err_type in enumerate(error_types):
                for c, arch in enumerate(architectures):
                    if len(error_types) > 1 and len(architectures) > 1:
                        ax = axes[r, c]
                    elif len(error_types) == 1:
                        ax = axes[c]
                    else:
                        ax = axes[r]

                    df_arch = df_subset[df_subset['Architecture'] == arch]
                    
                    if df_arch.empty:
                        continue
                        
                    r_id_val = df_arch['r_ideal'].iloc[0] if 'r_ideal' in df_arch and not df_arch['r_ideal'].empty else None
                    r_q_val = df_arch['r_quant'].iloc[0] if 'r_quant' in df_arch and not df_arch['r_quant'].empty else None
                    
                    coeff_str = ""
                    if r_id_val is not None and not pd.isna(r_id_val):
                        coeff_str += f"r_id: {r_id_val:.2f}"  # Ridotto a 2 decimali
                    if r_q_val is not None and not pd.isna(r_q_val):
                        if coeff_str: coeff_str += " | "
                        coeff_str += f"r_q: {r_q_val:.2f}"  # Ridotto a 2 decimali
                    
                    pivot_table = df_arch.pivot_table(values=err_type, index='Amplitude', columns='Frequency', aggfunc='mean')
                    
                    if pivot_table.empty:
                        continue
                        
                    sub_vals = pivot_table.values
                    sub_vals_pos = sub_vals[sub_vals > 0]
                    
                    if sub_vals_pos.size > 0:
                        vmin = sub_vals_pos.min()
                        vmax = sub_vals_pos.max()
                    else:
                        vmin, vmax = 1e-10, 1.0
                        
                    if vmin == vmax:
                        vmin = vmin / 10.0 if vmin > 0 else 1e-10
                        vmax = vmax * 10.0 if vmax != 0 else 1.0

                    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                        
                    sns.heatmap(
                        pivot_table, 
                        ax=ax, 
                        cmap='viridis', 
                        norm=norm,
                        cbar=True
                    )
                    
                    # --- APPLICAZIONE FORMATTAZIONE TICK (MAX 2 CIFRE DECIMALI) ---
                    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                    ax.set_xticklabels(
                        [f"{float(frequency):.2f}" for frequency in pivot_table.columns]
                    )
                    
                    # Ruota le etichette dell'asse X per evitare sovrapposizioni
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    # -------------------------------------------------------------
                    
                    title_text = f"{arch} | {err_type}\nBits: {bits_val}"
                    if coeff_str:
                        title_text += f" ({coeff_str})"
                    ax.set_title(title_text, fontsize=9)
                    
                    if r == len(error_types) - 1:
                        ax.set_xlabel("Frequency (Hz)")
                    else:
                        ax.set_xlabel("")
                        
                    if c == 0:
                        ax.set_ylabel("Amplitude")
                    else:
                        ax.set_ylabel("")

            plt.tight_layout()
            plt.savefig(Path(output_dir) / f"heatmap_{safe_bin_name}_bits_{bits_str_safe}.png", dpi=300, bbox_inches='tight')
            plt.close()
            time.sleep(0.1)


def heat_map_comparison(df_results, fy, output_dir="./images"):
    """Create a comparison grid with bit depths in rows and bin types in columns."""
    import matplotlib.ticker as ticker

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_types = [bin_type for bin_type in fy if bin_type in df_results["BinType"].values]
    architectures = list(df_results["Architecture"].dropna().unique())
    bit_levels = sorted(
        df_results["Bits"].unique(),
        key=lambda bits: (
            not str(bits).isdigit(),
            int(bits) if str(bits).isdigit() else str(bits),
        ),
    )
    error_types = ["Magnitude error", "Complex error"]

    if not bin_types or not architectures or not bit_levels:
        raise ValueError("No bin types, architectures, or bit levels available for heatmap comparison.")

    norms = {}
    for error_type in error_types:
        values = pd.to_numeric(df_results[error_type], errors="raise").to_numpy()
        positive_values = values[np.isfinite(values) & (values > 0)]
        if positive_values.size:
            vmin = float(positive_values.min())
            vmax = float(positive_values.max())
            if vmin == vmax:
                vmin /= 10
                vmax *= 10
        else:
            vmin, vmax = 1e-10, 1.0
        norms[error_type] = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    n_rows = len(bit_levels) * len(error_types)
    n_columns = len(bin_types) * len(architectures)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(2.5 * n_columns + 2, 1.9 * n_rows + 1.5),
        squeeze=False,
        layout="constrained",
    )
    mappables = {error_type: None for error_type in error_types}

    for bit_index, bits_val in enumerate(bit_levels):
        for error_index, error_type in enumerate(error_types):
            row = bit_index * len(error_types) + error_index
            for bin_index, bin_val in enumerate(bin_types):
                for arch_index, architecture in enumerate(architectures):
                    column = bin_index * len(architectures) + arch_index
                    ax = axes[row, column]
                    subset = df_results[
                        (df_results["Bits"] == bits_val)
                        & (df_results["BinType"] == bin_val)
                        & (df_results["Architecture"] == architecture)
                    ]
                    pivot = subset.pivot_table(
                        values=error_type,
                        index="Amplitude",
                        columns="Frequency",
                        aggfunc="mean",
                    )

                    if not pivot.empty:
                        mesh = sns.heatmap(
                            pivot,
                            ax=ax,
                            cmap="viridis",
                            norm=norms[error_type],
                            cbar=False,
                        )
                        mappables[error_type] = mesh.collections[0]
                        ax.set_xticklabels(
                            [f"{float(frequency):.2f}" for frequency in pivot.columns]
                        )
                        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                        plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=6)
                        ax.tick_params(axis="y", labelsize=6)

                    if row == 0:
                        ax.set_title(f"{bin_val}\n{architecture}", fontsize=8)
                    if column == 0:
                        ax.set_ylabel(f"Bits: {bits_val}\n{error_type}\nAmplitude", fontsize=8)
                    else:
                        ax.set_ylabel("")
                        ax.tick_params(axis="y", labelleft=False)
                    if row == n_rows - 1:
                        ax.set_xlabel("Frequency (Hz)", fontsize=7)
                    else:
                        ax.set_xlabel("")
                        ax.tick_params(axis="x", labelbottom=False)

    for error_index, error_type in enumerate(error_types):
        if mappables[error_type] is not None:
            error_axes = axes[error_index::len(error_types), :].ravel().tolist()
            fig.colorbar(
                mappables[error_type],
                ax=error_axes,
                fraction=0.015,
                pad=0.02,
                label=error_type,
            )

    fig.suptitle("Heatmap comparison by bit depth, bin type, architecture, and error", y=1.002)
    fig.savefig(
        output_dir / "heatmap_comparison_by_bits_and_bins.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)