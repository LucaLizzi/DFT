import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import time
import pandas as pd


def err_vs_freq(df_results, fy):
    """
    Error vs frequency plots
    """
    bin_types = list(fy.keys())
    bit_levels = df_results['Bits'].unique()
    
    for bin_val in bin_types:
        df_bin = df_results[df_results['BinType'] == bin_val]
        safe_bin_name = bin_val.replace("(", "_").replace(")", "").replace(".", "_")
        
        for bits_val in bit_levels:
            df_subset = df_bin[df_bin['Bits'] == bits_val]
            if df_subset.empty:
                continue
                
            bits_str_safe = str(bits_val).replace(" ", "_").replace("(", "").replace(")", "")
            
            # --- 1. MAGNITUDE ERROR VS FREQUENCY ---
            plt.figure(figsize=(10, 6))
            sns.lineplot(
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
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"./images/magnitude_error_freq_{safe_bin_name}_bits_{bits_str_safe}.png", dpi=300, bbox_inches='tight')
            plt.close()

            # --- 2. COMPLEX ERROR VS FREQUENCY ---
            plt.figure(figsize=(10, 6))
            sns.lineplot(
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
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"./images/complex_error_freq_{safe_bin_name}_bits_{bits_str_safe}.png", dpi=300, bbox_inches='tight')
            plt.close()


def bit_err(df_results, fy):
    """
    bits number vs error
    """
    bin_types = list(fy.keys())
    
    for bin_val in bin_types:
        df_bin = df_results[df_results['BinType'] == bin_val]
        
        df_quant = df_bin[df_bin['Bits'] != "Ideal (NoQuant)"].copy()
        df_quant['Bits'] = pd.to_numeric(df_quant['Bits'])
        
        df_ideal = df_bin[df_bin['Bits'] == "Ideal (NoQuant)"]
        
        if df_quant.empty:
            continue
            
        safe_name = bin_val.replace("(", "_").replace(")", "").replace(".", "_")
        
        # --- 1. MAGNITUDE ERROR VS BITS ---
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_quant, 
            x='Bits', 
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
            for arch, err_val in ideal_medians.items():
                color = arch_color_map.get(arch, 'black')
                plt.axhline(y=err_val, color=color, linestyle='--', linewidth=1.5, alpha=0.7, label=f"{arch} (Ideal)")

        plt.yscale('log')
        plt.minorticks_on()
        plt.gca().invert_xaxis()  
        plt.title(f"Magnitude Error vs Bit Variation ({bin_val})")
        plt.xlabel("Number of bits")
        plt.ylabel("Median Magnitude Error")
        plt.grid(True, which="major", ls="-", alpha=0.6)
        plt.grid(True, which="minor", ls=":", alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"./images/magnitude_error_vs_bits_number_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        time.sleep(0.1)

        # --- 2. COMPLEX ERROR VS BITS ---
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=df_quant, 
            x='Bits', 
            y='Complex error', 
            hue='Architecture', 
            estimator='median', 
            errorbar=('pi', 50), 
            marker='o'
        )
        
        if not df_ideal.empty:
            ideal_medians = df_ideal.groupby('Architecture')['Complex error'].median()
            for arch, err_val in ideal_medians.items():
                color = arch_color_map.get(arch, 'black')
                plt.axhline(y=err_val, color=color, linestyle='--', linewidth=1.5, alpha=0.7, label=f"{arch} (Ideal)")

        plt.yscale('log')
        plt.minorticks_on()
        plt.gca().invert_xaxis()  
        plt.title(f"Complex Error vs Bit Variation ({bin_val})")
        plt.xlabel("Number of bits")
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
    Heat map between amplitude, frequency and error
    """
    architectures = df_results['Architecture'].unique()
    bin_types = list(fy.keys())
    error_types = ['Magnitude error', 'Complex error']
    bit_levels = df_results['Bits'].unique()

    for bin_val in bin_types:
        safe_bin_name = bin_val.replace("(", "_").replace(")", "").replace(".", "_")
        df_bin = df_results[df_results['BinType'] == bin_val]
        
        for bits_val in bit_levels:
            df_subset = df_bin[df_bin['Bits'] == bits_val]
            if df_subset.empty:
                continue
                
            bits_str_safe = str(bits_val).replace(" ", "_").replace("(", "").replace(")", "")
            
            fig, axes = plt.subplots(len(error_types), len(architectures), 
                                     figsize=(len(architectures) * 4.5, 4 * len(error_types)), 
                                     sharex=True, sharey=True)

            if len(architectures) == 1:
                axes = axes[:, np.newaxis]

            for r, err_type in enumerate(error_types):
                sub_vals = df_subset[err_type]
                sub_vals_pos = sub_vals[sub_vals > 0]
                
                if sub_vals_pos.empty:
                    continue
                    
                vmin = sub_vals_pos.min()
                vmax = sub_vals_pos.max()
                
                if vmin == vmax:
                    vmin = vmin / 10.0 if vmin > 0 else 1e-10
                    vmax = vmax * 10.0 if vmax != 0 else 1.0

                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

                for c, arch in enumerate(architectures):
                    ax = axes[r, c]
                    df_arch = df_subset[df_subset['Architecture'] == arch]
                    
                    pivot_table = df_arch.pivot_table(values=err_type, index='Amplitude', columns='Frequency', aggfunc='mean')
                    
                    if pivot_table.empty:
                        continue
                        
                    sns.heatmap(
                        pivot_table, 
                        ax=ax, 
                        cmap='viridis', 
                        norm=norm,
                        cbar=True
                    )
                    
                    ax.set_title(f"{arch} | {err_type}\n(Bin: {bin_val} | Bits: {bits_val})")
                    
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