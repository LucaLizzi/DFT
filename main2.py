import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numba import jit
import matplotlib.animation as animation
from functions import _sdft_kernel, plt_func, quantizer
import time
import csv
import pandas as pd
import seaborn as sns


# ██╗   ██╗ █████╗ ██████╗ ██╗ █████╗ ██████╗ ██╗     ███████╗███████╗
# ██║   ██║██╔══██╗██╔══██╗██║██╔══██╗██╔══██╗██║     ██╔════╝██╔════╝
# ██║   ██║███████║██████╔╝██║███████║██████╔╝██║     █████╗  ███████╗
# ╚██╗ ██╔╝██╔══██║██╔══██╗██║██╔══██║██╔══██╗██║     ██╔══╝  ╚════██║
#  ╚████╔╝ ██║  ██║██║  ██║██║██║  ██║██████╔╝███████╗███████╗███████║
#   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝╚══════╝

# warm up the kernel
params = (True, False, False, False, False)
_, _, _, _, _ = _sdft_kernel(np.zeros(10, dtype=np.float64), 8, 30.0, 44100.0, params)
                                                                    
# sample_rate, data = wavfile.read("test3_notes.wav")
OSDFT_R = True  # resonator
OSDFT_M = True # modulator
RSDFT_DC = True# DC components
MSDFT_DC = True# DC components
DFT_FFT = True # FFT algorithm as ideality
params = (OSDFT_R, RSDFT_DC, OSDFT_M, MSDFT_DC, DFT_FFT)

window_size = 4096
fps_animation = 30
num_bits_list = [24, 8]
quantizer_enabler = [True]#, False]
max_amplitude = 100

# uncomment for test
fs = int(100e3)  #Hz
delta_f = fs / window_size

Ax = np.linspace(10/max_amplitude, 1, 3)       # normalized signal amplitudes
f_max = 15000
ky = np.unique(np.linspace(1, fs / (2 * delta_f), 3, dtype=int))
d = 0.25    # frequency sliding factor
fy  =   {"on_bin"  :   [k * delta_f  for k in ky],
        f"off_bin({d})"   :   [(k-d) * delta_f  for k in ky]}
        # f"off_bin({d*2})"   :   [(k-d*2) * delta_f  for k in ky],
        # f"off_bin({d*3})"   :   [(k-d*3) * delta_f  for k in ky]}    #Hz

T = 10  #min_periods / min_f
n = np.linspace(0, T, int(fs * T))


# ███████╗██████╗ ███████╗ ██████╗████████╗██████╗ ██╗   ██╗███╗   ███╗     █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗██╗███████╗
# ██╔════╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔══██╗██║   ██║████╗ ████║    ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝██╔════╝██║██╔════╝
# ███████╗██████╔╝█████╗  ██║        ██║   ██████╔╝██║   ██║██╔████╔██║    ███████║██╔██╗ ██║███████║██║   ╚████╔╝ ███████╗██║███████╗
# ╚════██║██╔═══╝ ██╔══╝  ██║        ██║   ██╔══██╗██║   ██║██║╚██╔╝██║    ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝  ╚════██║██║╚════██║
# ███████║██║     ███████╗╚██████╗   ██║   ██║  ██║╚██████╔╝██║ ╚═╝ ██║    ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████║██║███████║
# ╚══════╝╚═╝     ╚══════╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚═╝╚══════╝
                                                                                                                                            
# test 200 waves
beginning = time.time()

half_N = window_size // 2
err_rsdft_tot = np.zeros(half_N)
err_rsdft_dc_tot = np.zeros(half_N)
err_msdft_tot = np.zeros(half_N)
err_msdft_dc_tot = np.zeros(half_N)

results = []
# bin = [<bit number>, <amplitude>, <frequency index>]
selected_plots = {"on_bin"  :   [24, 0.1, 2],
                 f"off_bin({d})"  :   [24, 0.1, 2]}

with open('error_results.csv', mode='w', newline='', encoding='utf-8') as file_csv:
    writer = csv.writer(file_csv)

    writer.writerow(["wave name", "err rsdft", "err rsdft_dc", "err msdft", "err msdft_dc"])

    for quant_en in quantizer_enabler:
        for num_bits in num_bits_list:            
            for a in Ax:    
                for bin_, f_y in fy.items():
                    for index, f in enumerate(f_y):
                        wave_data = a * np.sin(2 * np.pi * f * n)
                        wave_name = f"amp={a}_f={f}Hz_{bin_}_{num_bits}"

                        start_time = time.time()

                        spectr_rsdft, spectr_rsdft_dc, spectr_msdft, spectr_msdft_dc, spectr_fft = _sdft_kernel(input_data=wave_data, 
                            N=window_size, fps_animation=fps_animation, sample_rate=fs, params=params, 
                            quant_en=quant_en, num_bits=num_bits, max_amplitude=max_amplitude)

                        # consider only the first half of the spectrum
                        spectr_rsdft = np.abs(spectr_rsdft[:half_N])
                        spectr_rsdft_dc = np.abs(spectr_rsdft_dc[:half_N])
                        spectr_msdft = np.abs(spectr_msdft[:half_N])
                        spectr_msdft_dc = np.abs(spectr_msdft_dc[:half_N])
                        spectr_fft = np.abs(spectr_fft[:half_N])

                        # error calculation
                        denum = np.linalg.norm(spectr_fft, ord=2)
                        err_rsdft = np.linalg.norm(spectr_rsdft - spectr_fft, ord=2) / denum
                        err_rsdft_dc = np.linalg.norm(spectr_rsdft_dc - spectr_fft, ord=2) / denum
                        err_msdft = np.linalg.norm(spectr_msdft - spectr_fft, ord=2) / denum
                        err_msdft_dc = np.linalg.norm(spectr_msdft_dc - spectr_fft, ord=2) / denum

                        results.append({
                            'Architecture': "rsdft", 'Bits': num_bits, 'Amplitude': a, 'BinType': bin_,
                            'Frequency': f, 'Error': err_rsdft})
                        results.append({
                            'Architecture': "rsdft_dc", 'Bits': num_bits, 'Amplitude': a, 'BinType': bin_,
                            'Frequency': f, 'Error': err_rsdft_dc})
                        results.append({
                            'Architecture': "msdft", 'Bits': num_bits, 'Amplitude': a, 'BinType': bin_,
                            'Frequency': f, 'Error': err_msdft})
                        results.append({
                            'Architecture': "msdft_dc", 'Bits': num_bits, 'Amplitude': a, 'BinType': bin_,
                            'Frequency': f, 'Error': err_msdft_dc})

                        # save plot spectrums and wave data
                        if selected_plots[bin_] == [num_bits, a, index]:
                            plt_func(f"{"quant" if quant_en else "no_quant"}_{wave_name}", wave_data, num_bits, max_amplitude,
                                spectr_rsdft, err_rsdft, 
                                spectr_rsdft_dc, err_rsdft_dc, 
                                spectr_msdft, err_msdft, 
                                spectr_msdft_dc, err_msdft_dc,
                                spectr_fft, range(half_N), fs/2)

                        print(f"Test finished: {wave_name}. \
                            Errors rsdft: {round(err_rsdft, 4)}, \
                            rsdft_dc: {round(err_rsdft_dc, 4)}, \
                            msdft: {round(err_msdft, 4)}, \
                            msdft_dc: {round(err_msdft_dc, 4)}. \
                            Execution time: {time.time() - start_time:.2f} seconds")

df_results = pd.DataFrame(results)

##################################################################################
                        # global error vs frequency
##################################################################################

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_results, 
    x='Frequency', 
    y='Error', 
    hue='Architecture', 
    style='BinType', 
    errorbar=None
)
plt.yscale('log') # Spesso la scala logaritmica aiuta a visualizzare errori di ordini di grandezza diversi
plt.title("global error vs frequency")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectral global error")
plt.grid(True, which="both", ls="--")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"./images/global_error_vs_frequency.png", dpi=300, bbox_inches='tight')
plt.show()
time.sleep(0.1)
plt.close()

##################################################################################
                        # error vs amplitude vs frequency
##################################################################################
architectures = df_results['Architecture'].unique()
vmin = df_results['Error'].min()
vmax = df_results['Error'].max()

fig, axes = plt.subplots(1, len(architectures), figsize=(18, 5), sharey=True)

for i, arch in enumerate(architectures):
    df_arch = df_results[df_results['Architecture'] == arch]
    # Heatmap
    pivot_table = df_arch.pivot_table(values='Error', index='Amplitude', columns='Frequency', aggfunc='mean')
    
    sns.heatmap(
        pivot_table, 
        ax=axes[i], 
        cmap='viridis', 
        vmin=vmin, 
        vmax=vmax, 
        cbar=(i == len(architectures)-1) # Shows the color bar only in the last plot
    )
    axes[i].set_title(f"Architecture: {arch}")
    axes[i].set_xlabel("Frequency (Hz)")
    if i == 0:
        axes[i].set_ylabel("Amplitude")

plt.suptitle("Error heatmap: amplitude vs frequency", y=1.03)
plt.tight_layout()
plt.savefig(f"./images/heatmap.png", dpi=300, bbox_inches='tight')
plt.show()
time.sleep(0.1)
plt.close()

##################################################################################
                        # bit vs error 
##################################################################################
plt.figure(figsize=(8, 5))
sns.lineplot(
    data=df_results, 
    x='Bits', 
    y='Error', 
    hue='Architecture', 
    estimator='median',   # Usa la mediana come richiesto
    errorbar=('pi', 50),  # Intervallo di percentile (es. 50% centrale)
    marker='o'
)
plt.yscale('log')
plt.gca().invert_xaxis() # I bit solitamente vanno da alti a bassi (es. 64 a 2)
plt.title("Error vs bit variation")
plt.xlabel("Number of bits")
plt.ylabel("Median error (with percentiles)")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.savefig(f"./images/error_vs_bits_number.png", dpi=300, bbox_inches='tight')
plt.show()
time.sleep(0.1)
plt.close()

end_time = time.time()
print(f"Test finished. Total time: {end_time - beginning:.2f} seconds")

print("=" * 15 + " RESULTS " + "=" * 15)
print("=" * 40)