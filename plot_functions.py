import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import time
import pandas as pd


def err_vs_freq(df_results, fy):
    bin_types = list(fy.keys())
    bit_levels = df_results['Bits'].unique()
    
    for bin_val in bin_types:
        df_bin = df_results[df_results['BinType'] == bin_val]
        safe_bin_name = str(bin_val).replace("(", "_").replace(")", "").replace(".", "_")
        
        for bits_val in bit_levels:
            df_subset = df_bin[df_bin['Bits'] == bits_val]
            if df_subset.empty:
                continue
                
            bits_str_safe = str(bits_val).replace(" ", "_").replace("(", "").replace(")", "")
            
            legend_labels = {}
            for arch in df_subset['Architecture'].unique():
                df_arch = df_subset[df_subset['Architecture'] == arch]
                r_val = None
                
                # Se siamo nel caso ideale, usiamo r_ideal; altrimenti r_quant (se disponibile)
                if bits_val == "Ideal (NoQuant)":
                    if 'r_ideal' in df_arch and not df_arch['r_ideal'].empty and not pd.isna(df_arch['r_ideal'].iloc[0]):
                        r_val = df_arch['r_ideal'].iloc[0]
                else:
                    if 'r_quant' in df_arch and not df_arch['r_quant'].empty and not pd.isna(df_arch['r_quant'].iloc[0]):
                        r_val = df_arch['r_quant'].iloc[0]
                    elif 'r_ideal' in df_arch and not df_arch['r_ideal'].empty and not pd.isna(df_arch['r_ideal'].iloc[0]):
                        r_val = df_arch['r_ideal'].iloc[0]
                
                if r_val is not None:
                    legend_labels[arch] = f"{arch} (r={r_val:.4f})"
                else:
                    legend_labels[arch] = arch

            # --- 1. MAGNITUDE ERROR VS FREQUENCY ---
            plt.figure(figsize=(10, 6))
            ax = sns.lineplot(
                data=df_subset, 
                x='Frequency', 
                y='Magnitude error', 
                hue='Architecture', 
                errorbar=None
            )  
            plt.yscale('log')
            plt.minorticks_on()
            plt.title(f"Magnitude Error vs Frequency\nBin: {bin_val} | Bits: {bits_val}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Spectral Magnitude Error")
            plt.grid(True, which="major", ls="-", alpha=0.6)
            plt.grid(True, which="minor", ls=":", alpha=0.3)
            
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [legend_labels.get(lbl, lbl) for lbl in labels]
            plt.legend(handles=handles, labels=new_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(f"./images/magnitude_error_freq_{safe_bin_name}_bits_{bits_str_safe}.png", dpi=300, bbox_inches='tight')
            plt.close()

            # --- 2. COMPLEX ERROR VS FREQUENCY ---
            plt.figure(figsize=(10, 6))
            ax = sns.lineplot(
                data=df_subset, 
                x='Frequency', 
                y='Complex error', 
                hue='Architecture', 
                errorbar=None
            )  
            plt.yscale('log')
            plt.minorticks_on()
            plt.title(f"Complex Global Error vs Frequency\nBin: {bin_val} | Bits: {bits_val}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Spectral Complex Error")
            plt.grid(True, which="major", ls="-", alpha=0.6)
            plt.grid(True, which="minor", ls=":", alpha=0.3)
            
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [legend_labels.get(lbl, lbl) for lbl in labels]
            plt.legend(handles=handles, labels=new_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(f"./images/complex_error_freq_{safe_bin_name}_bits_{bits_str_safe}.png", dpi=300, bbox_inches='tight')
            plt.close()


def bit_err(df_results, fy):
    bin_types = list(fy.keys())
    
    for bin_val in bin_types:
        df_bin = df_results[df_results['BinType'] == bin_val]
        
        df_quant = df_bin[df_bin['Bits'] != "Ideal (NoQuant)"].copy()
        df_quant['Bits_num'] = pd.to_numeric(df_quant['Bits'])
        
        df_ideal = df_bin[df_bin['Bits'] == "Ideal (NoQuant)"]
        
        if df_quant.empty:
            continue
            
        safe_name = str(bin_val).replace("(", "_").replace(")", "").replace(".", "_")
        
        r_quant_by_bits = df_quant.groupby('Bits_num')['r_quant'].mean()
        bit_labels_map = {
            b: f"{b}\n(r_q={r_quant_by_bits[b]:.4f})" if b in r_quant_by_bits and not pd.isna(r_quant_by_bits[b]) else str(b)
            for b in sorted(df_quant['Bits_num'].unique())
        }
        
        # --- 1. MAGNITUDE ERROR VS BITS ---
        plt.figure(figsize=(9, 5.5))
        ax = sns.lineplot(
            data=df_quant, 
            x='Bits_num', 
            y='Magnitude error', 
            hue='Architecture', 
            estimator='median', 
            errorbar=('pi', 50), 
            marker='o'
        )
        
        palette = sns.color_palette()
        architectures = df_quant['Architecture'].unique()
        arch_color_map = {arch: palette[i] for i, arch in enumerate(architectures)}
        
        if not df_ideal.empty:
            ideal_medians = df_ideal.groupby('Architecture')['Magnitude error'].median()
            r_ideal_medians = df_ideal.groupby('Architecture')['r_ideal'].first()
            
            for arch, err_val in ideal_medians.items():
                color = arch_color_map.get(arch, 'black')
                r_id_str = f" (r_ideal={r_ideal_medians[arch]:.4f})" if arch in r_ideal_medians and not pd.isna(r_ideal_medians[arch]) else ""
                plt.axhline(
                    y=err_val, 
                    color=color, 
                    linestyle='--', 
                    linewidth=1.5, 
                    alpha=0.7, 
                    label=f"{arch} Ideal{r_id_str}"
                )

        plt.yscale('log')
        plt.minorticks_on()
        plt.gca().invert_xaxis()
        
        unique_bits = sorted(df_quant['Bits_num'].unique(), reverse=True)
        plt.xticks(unique_bits, [bit_labels_map[b] for b in unique_bits])
        
        plt.title(f"Magnitude Error vs Bit Variation ({bin_val})")
        plt.xlabel("Number of Bits (with r_quant)")
        plt.ylabel("Median Magnitude Error")
        plt.grid(True, which="major", ls="-", alpha=0.6)
        plt.grid(True, which="minor", ls=":", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"./images/magnitude_error_vs_bits_number_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        time.sleep(0.1)

        # --- 2. COMPLEX ERROR VS BITS ---
        plt.figure(figsize=(9, 5.5))
        ax = sns.lineplot(
            data=df_quant, 
            x='Bits_num', 
            y='Complex error', 
            hue='Architecture', 
            estimator='median', 
            errorbar=('pi', 50), 
            marker='o'
        )
        
        if not df_ideal.empty:
            ideal_medians = df_ideal.groupby('Architecture')['Complex error'].median()
            r_ideal_medians = df_ideal.groupby('Architecture')['r_ideal'].first()
            
            for arch, err_val in ideal_medians.items():
                color = arch_color_map.get(arch, 'black')
                r_id_str = f" (r_ideal={r_ideal_medians[arch]:.4f})" if arch in r_ideal_medians and not pd.isna(r_ideal_medians[arch]) else ""
                plt.axhline(
                    y=err_val, 
                    color=color, 
                    linestyle='--', 
                    linewidth=1.5, 
                    alpha=0.7, 
                    label=f"{arch} Ideal{r_id_str}"
                )

        plt.yscale('log')
        plt.minorticks_on()
        plt.gca().invert_xaxis()  
        
        plt.xticks(unique_bits, [bit_labels_map[b] for b in unique_bits])
        
        plt.title(f"Complex Error vs Bit Variation ({bin_val})")
        plt.xlabel("Number of Bits (with r_quant)")
        plt.ylabel("Median Complex Error")
        plt.grid(True, which="major", ls="-", alpha=0.6)
        plt.grid(True, which="minor", ls=":", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"./images/complex_error_vs_bits_number_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        time.sleep(0.1)


def heat_map(df_results, fy):
    """
    Heat map between amplitude, frequency and error, including r_ideal and r_quant in titles.
    """
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
                        coeff_str += f"r_id: {r_id_val:.4f}"
                    if r_q_val is not None and not pd.isna(r_q_val):
                        if coeff_str: coeff_str += " | "
                        coeff_str += f"r_q: {r_q_val:.4f}"
                    
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
            plt.savefig(f"./images/heatmap_{safe_bin_name}_bits_{bits_str_safe}.png", dpi=300, bbox_inches='tight')
            plt.close()
            time.sleep(0.1)