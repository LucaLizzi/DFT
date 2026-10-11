import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numba import jit
import matplotlib.animation as animation
from functions import _sdft_kernel
import time
import csv


# ██╗   ██╗ █████╗ ██████╗ ██╗ █████╗ ██████╗ ██╗     ███████╗███████╗
# ██║   ██║██╔══██╗██╔══██╗██║██╔══██╗██╔══██╗██║     ██╔════╝██╔════╝
# ██║   ██║███████║██████╔╝██║███████║██████╔╝██║     █████╗  ███████╗
# ╚██╗ ██╔╝██╔══██║██╔══██╗██║██╔══██║██╔══██╗██║     ██╔══╝  ╚════██║
#  ╚████╔╝ ██║  ██║██║  ██║██║██║  ██║██████╔╝███████╗███████╗███████║
#   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝╚══════╝

# warm up the kernel
params = (True, False, False, False)
_, _, _, _ = _sdft_kernel(np.zeros(10, dtype=np.float64), 8, 30.0, 44100.0, params)
                                                                    
# sample_rate, data = wavfile.read("test3_notes.wav")
OSDFT_R = True  # resonator
OSDFT_M = True # modulator
OSDFT_DC = True# DC components
DFT_FFT = True # FFT algorithm as ideality
params = (OSDFT_R, OSDFT_DC, OSDFT_M, DFT_FFT)

# uncomment for test
fs = int(40e3)  #Hz
# n = np.linspace(0, 1, fs)
# N = len(n)  # number of samples
Ax = np.linspace(10, 100, 10)
ky = np.linspace(1, 15000, 15)
d = 0.25    # frequency sliding factor
fy  =   {"on_bin"  :   [k  for k in ky],
        "off_bin"   :   [(k-d)  for k in ky]}    #Hz

min_f = min(min(fy["on_bin"]), min(fy["off_bin"]))
min_periods = 1
T = min_periods / min_f
n = np.linspace(0, T, int(fs * T))

s_xy = {}   # test dict, 200 waves
for a in Ax:    
    for bin, f_y in fy.items():
        for f in f_y:
            s_xy[f"test_{a}_{f}Hz_{bin}"] = a * np.sin(2 * np.pi * f * n)

window_size = 4096
fps_animation = 30


# ███████╗██████╗ ███████╗ ██████╗████████╗██████╗ ██╗   ██╗███╗   ███╗     █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗██╗███████╗
# ██╔════╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔══██╗██║   ██║████╗ ████║    ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝██╔════╝██║██╔════╝
# ███████╗██████╔╝█████╗  ██║        ██║   ██████╔╝██║   ██║██╔████╔██║    ███████║██╔██╗ ██║███████║██║   ╚████╔╝ ███████╗██║███████╗
# ╚════██║██╔═══╝ ██╔══╝  ██║        ██║   ██╔══██╗██║   ██║██║╚██╔╝██║    ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝  ╚════██║██║╚════██║
# ███████║██║     ███████╗╚██████╗   ██║   ██║  ██║╚██████╔╝██║ ╚═╝ ██║    ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████║██║███████║
# ╚══════╝╚═╝     ╚══════╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚═╝╚══════╝
                                                                                                                                            
# test 200 waves
print("Test started for all waves")
beginning = time.time()

half_N = window_size // 2
err_rsdft_tot = np.zeros(half_N)
err_rsdft_dc_tot = np.zeros(half_N)
err_msdft_tot = np.zeros(half_N)

with open('error_results.csv', mode='w', newline='', encoding='utf-8') as file_csv:
    writer = csv.writer(file_csv)

    writer.writerow(["wave name", "err rsdft", "err rsdft_dc", "err msdft"])

    for wave_name, wave_data in s_xy.items():
        start_time = time.time()

        spectr_rsdft, spectr_rsdft_dc, spectr_msdft, spectr_fft = _sdft_kernel(wave_data, window_size, fps_animation, fs, params)

        # consider only the first half of the spectrum
        spectr_rsdft = np.abs(spectr_rsdft[:half_N])
        spectr_rsdft_dc = np.abs(spectr_rsdft_dc[:half_N])
        spectr_msdft = np.abs(spectr_msdft[:half_N])
        spectr_fft = np.abs(spectr_fft[:half_N])

        # error calculation
        denum = np.sqrt(spectr_fft**2)
        err_rsdft = np.sqrt((spectr_rsdft - spectr_fft)**2) / denum
        err_rsdft_dc = np.sqrt((spectr_rsdft_dc - spectr_fft)**2) / denum
        err_msdft = np.sqrt((spectr_msdft - spectr_fft)**2) / denum

        err_rsdft_tot += err_rsdft
        err_rsdft_dc_tot += err_rsdft_dc
        err_msdft_tot += err_msdft

        writer.writerow([wave_name, 
            round(np.mean(err_rsdft), 2), 
            round(np.mean(err_rsdft_dc), 2), 
            round(np.mean(err_msdft), 2)])

        print(f"Test finished: {wave_name}. \
            Errors rsdft: {round(np.mean(err_rsdft), 2)}, \
            rsdft_dc: {round(np.mean(err_rsdft_dc), 2)}, \
            msdft: {round(np.mean(err_msdft), 2)}. \
            Execution time: {time.time() - start_time:.2f} seconds")

err_rsdft_avg = err_rsdft_tot / len(s_xy)
err_rsdft_dc_avg = err_rsdft_dc_tot / len(s_xy)
err_msdft_avg = err_msdft_tot / len(s_xy)

end_time = time.time()
print(f"Test finished. Total time: {end_time - beginning:.2f} seconds")

print("=" * 15 + " RESULTS " + "=" * 15)
print(f"Average error rsdft: {round(np.mean(err_rsdft_avg), 2)}")
print(f"Average error rsdft_dc: {round(np.mean(err_rsdft_dc_avg), 2)}")
print(f"Average error msdft: {round(np.mean(err_msdft_avg), 2)}")
print("=" * 40)