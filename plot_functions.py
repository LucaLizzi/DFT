import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

def err_vs_freq(df_results):
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_results, 
        x='Frequency', 
        y='Magnitude error', 
        hue='Architecture', 
        style='BinType', 
        errorbar=None
    )  
    plt.yscale('log')
    plt.title("Magnitude Error vs Frequency")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Spectral Magnitude Error")
    plt.grid(True, which="both", ls="--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("./images/magnitude_error_vs_frequency.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # --- 2. Grafico Complex Error ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_results, 
        x='Frequency', 
        y='Complex error', 
        hue='Architecture', 
        style='BinType', 
        errorbar=None
    )  
    plt.yscale('log')
    plt.title("Complex Global Error vs Frequency")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Spectral Complex Error")
    plt.grid(True, which="both", ls="--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("./images/complex_error_vs_frequency.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def heat_map(df_results):
    architectures = df_results['Architecture'].unique()
    error_types = ['Magnitude error', 'Complex error']

    # Creiamo una griglia: 2 righe (i tipi di errore) x N colonne (le architetture)
    fig, axes = plt.subplots(len(error_types), len(architectures), figsize=(len(architectures) * 5, 4 * len(error_types)), sharex=True, sharey=True)

    # Gestione del caso in cui ci sia una sola architettura
    if len(architectures) == 1:
        axes = axes[:, np.newaxis]

    for r, err_type in enumerate(error_types):
        # Calcoliamo vmin e vmax specifici per ogni metrica per sfruttare al meglio la scala di colori
        vmin = df_results[err_type].min()
        vmax = df_results[err_type].max()
        
        for c, arch in enumerate(architectures):
            ax = axes[r, c]
            df_arch = df_results[df_results['Architecture'] == arch]
            
            # Pivot table basata sul tipo di errore corrente
            pivot_table = df_arch.pivot_table(values=err_type, index='Amplitude', columns='Frequency', aggfunc='mean')
            
            sns.heatmap(
                pivot_table, 
                ax=ax, 
                cmap='viridis', 
                vmin=vmin, 
                vmax=vmax, 
                cbar=(c == len(architectures) - 1) # Mostra la colorbar solo sull'ultima colonna di ogni riga
            )
            
            ax.set_title(f"{arch} - {err_type}")
            
            # Etichette assi pulite solo sui bordi esterni
            if r == len(error_types) - 1:
                ax.set_xlabel("Frequency (Hz)")
            else:
                ax.set_xlabel("")
                
            if c == 0:
                ax.set_ylabel("Amplitude")
            else:
                ax.set_ylabel("")

    plt.suptitle("Error heatmaps: Amplitude vs Frequency (Magnitude & Complex)", y=1.02)
    plt.tight_layout()
    plt.savefig("./images/heatmaps_magnitude_and_complex.png", dpi=300, bbox_inches='tight')
    plt.show()
    time.sleep(0.1)
    plt.close()


def bit_err(df_results):
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_results, 
        x='Bits', 
        y='Magnitude error', 
        hue='Architecture', 
        estimator='median',   # Usa la mediana
        errorbar=('pi', 50),  # Intervallo di percentile (50% centrale)
        marker='o'
    )
    plt.yscale('log')
    plt.gca().invert_xaxis()  # I bit vanno da alti a bassi (es. 24 a 8)
    plt.title("Magnitude Error vs Bit Variation")
    plt.xlabel("Number of bits")
    plt.ylabel("Median Magnitude Error")
    plt.grid(True, which="both", ls="--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("./images/magnitude_error_vs_bits_number.png", dpi=300, bbox_inches='tight')
    plt.show()
    time.sleep(0.1)
    plt.close()


    # --- 2. Grafico per il Complex Error vs Bits ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_results, 
        x='Bits', 
        y='Complex error', 
        hue='Architecture', 
        estimator='median',   # Usa la mediana
        errorbar=('pi', 50),  # Intervallo di percentile (50% centrale)
        marker='o'
    )
    plt.yscale('log')
    plt.gca().invert_xaxis()  # I bit vanno da alti a bassi
    plt.title("Complex Error vs Bit Variation")
    plt.xlabel("Number of bits")
    plt.ylabel("Median Complex Error")
    plt.grid(True, which="both", ls="--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("./images/complex_error_vs_bits_number.png", dpi=300, bbox_inches='tight')
    plt.show()
    time.sleep(0.1)
    plt.close()

