import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numba import jit
import matplotlib.animation as animation
from functions import _sdft_kernel, plt_func, quantizer
from plot_functions import heat_map, err_vs_freq, bit_err
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
num_bits_list = [64, 32, 16, 8]
quantizer_enabler = [False, True]
max_amplitude = 100
f_max = 15000
fs = int(40e3)  #Hz
delta_f = fs / window_size

min_k = 1
max_k = min(window_size // 2 - 1, int(f_max / delta_f))     # ignore Nyquist and DC bins  
ky = np.unique(np.linspace(min_k, max_k, 10, dtype=int))
d = 0.25    # frequency sliding factor
fy  =   {"on_bin"  :   [k * delta_f  for k in ky],
        f"off_bin({d})"   :   [(k-d) * delta_f  for k in ky]}   #Hz

Ax = np.linspace(10/max_amplitude, 1, 10)       # normalized signal amplitudes

n = np.arange(window_size)/fs


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
# bin = [<bit number>, <amplitude index>, <frequency index>, <onbin/offbin>, <quant_en>]
selected_plots = [(8, 3, 5, 0, False), 
                (8, 8, 5, 0, False), 
                (8, 3, 5, d, False),
                (8, 3, 5, d, False),
                (8, 8, 5, 0, True), 
                (8, 8, 5, 0, True), 
                (8, 3, 5, d, True),
                (8, 8, 5, d, True),
                (32, 3, 5, 0, True), 
                (32, 8, 5, 0, True), 
                (32, 3, 5, d, True),
                (32, 8, 5, d, True)]

with open('error_results.csv', mode='w', newline='', encoding='utf-8') as file_csv:
    writer = csv.writer(file_csv)

    writer.writerow(["wave name", "err absolute rsdft", "err complex rsdft", "err absolute rsdft_dc", "err complex rsdft_dc", 
                    "err absolute msdft", "err complex msdft", "err absolute msdft_dc", "err complex msdft_dc"])

    for quant_en in quantizer_enabler:
        for num_bits in num_bits_list:            
            num_bits = num_bits if quant_en else "Ideal (NoQuant)"

            for index_a, a in enumerate(Ax):    
                for bin_, f_y in fy.items():
                    for index_f, f in enumerate(f_y):
                        if quant_en:
                            wave_data = quantizer(a * np.sin(2 * np.pi * f * n), num_bits=num_bits)
                            wave_name = f"amp={round(a,2)}_f={round(f,2)}Hz_{bin_}_{num_bits}bits"
                        else:   
                            wave_data = a * np.sin(2 * np.pi * f * n)
                            wave_name = f"amp={a}_f={f}Hz_{bin_}_noQuant"

                        start_time = time.time()

                        spectr_rsdft, spectr_rsdft_dc, spectr_msdft, spectr_msdft_dc, spectr_fft = _sdft_kernel(input_data=wave_data, 
                            N=window_size, fps_animation=fps_animation, sample_rate=fs, params=params, 
                            quant_en=quant_en, num_bits=num_bits, max_amplitude=max_amplitude)

                        # consider only the first half of the spectrum
                        spectr_rsdft = spectr_rsdft[:half_N]
                        spectr_rsdft_dc = spectr_rsdft_dc[:half_N]
                        spectr_msdft = spectr_msdft[:half_N]
                        spectr_msdft_dc = spectr_msdft_dc[:half_N]
                        spectr_fft = spectr_fft[:half_N]

                        # complex error
                        denum = np.linalg.norm(spectr_fft, ord=2)
                        err_cpx_rsdft = np.linalg.norm(spectr_rsdft - spectr_fft, ord=2) / denum
                        err_cpx_rsdft_dc = np.linalg.norm(spectr_rsdft_dc - spectr_fft, ord=2) / denum
                        err_cpx_msdft = np.linalg.norm(spectr_msdft - spectr_fft, ord=2) / denum
                        err_cpx_msdft_dc = np.linalg.norm(spectr_msdft_dc - spectr_fft, ord=2) / denum

                        # absolute values for magnitude error
                        spectr_rsdft = np.abs(spectr_rsdft)
                        spectr_rsdft_dc = np.abs(spectr_rsdft_dc)
                        spectr_msdft = np.abs(spectr_msdft)
                        spectr_msdft_dc = np.abs(spectr_msdft_dc)
                        spectr_fft = np.abs(spectr_fft)

                        # magnitude error
                        denum = np.linalg.norm(spectr_fft, ord=2)
                        err_abs_rsdft = np.linalg.norm(spectr_rsdft - spectr_fft, ord=2) / denum
                        err_abs_rsdft_dc = np.linalg.norm(spectr_rsdft_dc - spectr_fft, ord=2) / denum
                        err_abs_msdft = np.linalg.norm(spectr_msdft - spectr_fft, ord=2) / denum
                        err_abs_msdft_dc = np.linalg.norm(spectr_msdft_dc - spectr_fft, ord=2) / denum

                        # save results
                        results.append({
                            'Architecture': "rsdft", 'Bits': num_bits, 'Amplitude': a, 'BinType': bin_,
                            'Frequency': f, 'Magnitude error': err_abs_rsdft, "Complex error": err_cpx_rsdft})
                        results.append({
                            'Architecture': "rsdft_dc", 'Bits': num_bits, 'Amplitude': a, 'BinType': bin_,
                            'Frequency': f, 'Magnitude error': err_abs_rsdft_dc, "Complex error": err_cpx_rsdft_dc})
                        results.append({
                            'Architecture': "msdft", 'Bits': num_bits, 'Amplitude': a, 'BinType': bin_,
                            'Frequency': f, 'Magnitude error': err_abs_msdft, "Complex error": err_cpx_msdft})
                        results.append({
                            'Architecture': "msdft_dc", 'Bits': num_bits, 'Amplitude': a, 'BinType': bin_,
                            'Frequency': f, 'Magnitude error': err_abs_msdft_dc, "Complex error": err_cpx_msdft_dc})

                        # save plot spectrums and wave data
                        if (num_bits, index_a, index_f, 0 if bin_ == "on_bin" else d, quant_en) in selected_plots or "ALL" in selected_plots:
                            plt_func(f"{"quant" if quant_en else "no_quant"}_{wave_name}", wave_data, num_bits, max_amplitude,
                                spectr_rsdft, spectr_rsdft_dc, spectr_msdft, spectr_msdft_dc, spectr_fft, range(half_N), fs/2)

                        writer.writerow([
                            wave_name, 
                            err_abs_rsdft, err_cpx_rsdft, 
                            err_abs_rsdft_dc, err_cpx_rsdft_dc, 
                            err_abs_msdft, err_cpx_msdft, 
                            err_abs_msdft_dc, err_cpx_msdft_dc
                        ])

                        print(f"Test finished: {wave_name}. Errors: \n" \
                                f"  RSDFT     -> magnitude: {round(err_abs_rsdft, 4)}, complex: {round(err_cpx_rsdft, 4)} \n"
                                f"  RSDFT_DC  -> magnitude: {round(err_abs_rsdft_dc, 4)}, complex: {round(err_cpx_rsdft_dc, 4)} \n"
                                f"  MSDFT     -> magnitude: {round(err_abs_msdft, 4)}, complex: {round(err_cpx_msdft, 4)} \n"
                                f"  MSDFT_DC  -> magnitude: {round(err_abs_msdft_dc, 4)}, complex: {round(err_cpx_msdft_dc, 4)} \n"
                                f"  Execution time: {time.time() - start_time:.2f} seconds \n" \
                                f"================================================================================")

            if quant_en == False:    break


df_results = pd.DataFrame(results)

bit_err(df_results, fy=fy)
err_vs_freq(df_results, fy=fy)
heat_map(df_results, fy=fy)

end_time = time.time()
print(f"Test finished. Total time: {end_time - beginning:.2f} seconds")
