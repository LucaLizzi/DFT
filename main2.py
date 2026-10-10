import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numba import jit
import matplotlib.animation as animation
from functions import _sdft_kernel


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
ky = np.linspace(1, 10000, 10)
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
wave_name = list(s_xy.keys())[9]
wave_data = s_xy[wave_name]
print(f"Test started: {wave_name}")

spectr_rsdft, spectr_rsdft_dc, spectr_msdft, spectr_fft = _sdft_kernel(wave_data, window_size, fps_animation, fs, params)



# for wave_name, wave_data in s_xy.items():
#     # resonator branch oSDFT
#     _, _, _, _ = _sdft_kernel(np.zeros(10, dtype=np.float64), 8, 30.0, 44100.0, params)


print("Test finished")



#  █████╗ ███╗   ██╗██╗███╗   ███╗ █████╗ ████████╗███████╗██████╗     ██████╗ ██╗      ██████╗ ████████╗
# ██╔══██╗████╗  ██║██║████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██╔══██╗    ██╔══██╗██║     ██╔═══██╗╚══██╔══╝
# ███████║██╔██╗ ██║██║██╔████╔██║███████║   ██║   █████╗  ██║  ██║    ██████╔╝██║     ██║   ██║   ██║   
# ██╔══██║██║╚██╗██║██║██║╚██╔╝██║██╔══██║   ██║   ██╔══╝  ██║  ██║    ██╔═══╝ ██║     ██║   ██║   ██║   
# ██║  ██║██║ ╚████║██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗██████╔╝    ██║     ███████╗╚██████╔╝   ██║   
# ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═════╝     ╚═╝     ╚══════╝ ╚═════╝    ╚═╝   
                                                                                                       
print("Plotting stage started")
# (help from Gemini)
# avoid negative spectrum
half_N = window_size // 2
freq_axis = np.linspace(0, fs / 2, half_N)
freq_axis_dB = 20 * np.log10(freq_axis + 1e-12)
# spectr_rsdft_dB = np.zeros(half_N)
# spectr_rsdft_dc_dB = np.zeros(half_N)
# spectr_msdft_dB = np.zeros(half_N)
# spectr_fft_dB = np.zeros(half_N)

if OSDFT_R:
    spectr_rsdft = np.abs(spectr_rsdft[:half_N])# / (window_size / 2)
    # spectr_rsdft_dB = 20 * np.log10(_ + 1e-12)

if OSDFT_DC:
    spectr_rsdft_dc = np.abs(spectr_rsdft_dc[:half_N])# / (window_size / 2)
    # spectr_rsdft_dc_dB = 20 * np.log10(_ + 1e-12)

if OSDFT_M:
    spectr_msdft = np.abs(spectr_msdft[:half_N])# / (window_size / 2)
    # spectr_msdft_dB = 20 * np.log10(_ + 1e-12)

if DFT_FFT:
    spectr_fft = np.abs(spectr_fft[:half_N])# / (window_size / 2)
    # spectr_fft_dB = 20 * np.log10(_ + 1e-12)

denum = np.sqrt(spectr_fft**2)
err_rsdft = np.sqrt((spectr_rsdft - spectr_fft)**2) / denum
err_rsdft_dc = np.sqrt((spectr_rsdft_dc - spectr_fft)**2) / denum
err_msdft = np.sqrt((spectr_msdft - spectr_fft)**2) / denum

err_rsdft_time = np.real(np.fft.ifft(err_rsdft))
# err_rsdft_time = err_rsdft_time[len(n)//2 - 100:len(n)//2 + 100]
err_rsdft_dc_time = np.real(np.fft.ifft(err_rsdft_dc))
# err_rsdft_dc_time = err_rsdft_dc_time[len(n)//2 - 100:len(n)//2 + 100]
err_msdft_time = np.real(np.fft.ifft(err_msdft))
# err_msdft_time = err_msdft_time[len(n)//2 - 100:len(n)//2 + 100]

fig, ax = plt.subplots(3, 3, figsize=(10, 7))

ax[0, 0].plot(n, wave_data, color='gray', alpha=0.6)
ax[0, 0].set_title("Time signal")
if DFT_FFT:
    ax[0, 1].plot(freq_axis, spectr_fft, color='gray', alpha=0.6)
    ax[0, 1].set_title("fft")
    ax[0, 1].set_xscale('log')
if OSDFT_R:
    ax[1, 0].plot(freq_axis, spectr_rsdft, color='gray', alpha=0.6)
    ax[1, 0].set_title("rsdft")
    ax[1, 0].set_xscale('log')
    # ax[2, 0].plot(freq_axis, err_rsdft, color='red', alpha=0.6)
    ax[2, 0].plot(range(len(err_rsdft_time)), err_rsdft_time, color='red', alpha=0.6)
    ax[2, 0].set_title("err_rsdft")
if OSDFT_DC:
    ax[1, 1].plot(freq_axis, spectr_rsdft_dc, color='gray', alpha=0.6)
    ax[1, 1].set_title("rsdft_dc")
    ax[1, 1].set_xscale('log')
    # ax[2, 1].plot(freq_axis, err_rsdft_dc, color='red', alpha=0.6)
    ax[2, 1].plot(range(len(err_rsdft_dc_time)), err_rsdft_dc_time, color='red', alpha=0.6)
    ax[2, 1].set_title("err_rsdft_dc")
if OSDFT_M:
    ax[1, 2].plot(freq_axis, spectr_msdft, color='gray', alpha=0.6)
    ax[1, 2].set_title("msdft")
    ax[1, 2].set_xscale('log')
    # ax[2, 2].plot(freq_axis, err_msdft, color='red', alpha=0.6)
    ax[2, 2].plot(range(len(err_msdft_time)), err_msdft_time, color='red', alpha=0.6)
    ax[2, 2].set_title("err_msdft")


plt.tight_layout()
plt.show()

# # takes in account the first half_N samples of the spectrum
# spectr_mag = np.abs(spectr_fps[:, :half_N]) / (window_size / 2)
# num_frames = spectr_mag.shape[0]
# spectr_db = 20 * np.log10(spectr_mag + 1e-12)   #1e-12 needed to avoid inf values


# fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 7))

# # --- 1. TIME PLOT ---
# x_time = np.linspace(0, audio_seconds, data.size)
# ax_time.plot(x_time, data, color='gray', alpha=0.6)
# line_cursor = ax_time.axvline(x=0, color='red', linestyle='--')

# ax_time.set_title("Time signal")
# ax_time.set_xlabel("Time (s)")
# ax_time.set_ylabel("Amplitude")
# ax_time.set_xlim(0, audio_seconds)
# ax_time.grid(True)

# # --- 2. FREQUENCY PLOT ---
# (line_freq,) = ax_freq.plot(freq_axis, np.full(half_N, -120.0), color="crimson", lw=1.5)

# ax_freq.set_title("Spectrum SDFT")
# ax_freq.set_xlabel("Frequency (Hz)")
# ax_freq.set_ylabel("Amplitude (dB)")
# ax_freq.grid(True)
# # ax_freq.set_xlim(0, 2000)  # uncomment to zoom in

# ax_freq.set_xscale('log')

# max_db = np.max(spectr_db)
# ax_freq.set_ylim(-120, max(max_db + 5, 0))
# ax_freq.grid(True)

# plt.tight_layout()

# # --- 3. ANIMATION ---
# def update(i):
#     time_now = i * (audio_seconds / num_frames)

#     line_cursor.set_xdata([time_now])

#     line_freq.set_ydata(spectr_db[i])

#     ax_freq.set_title(f"Spettro SDFT a t = {time_now:.2f} s")

#     return line_cursor, line_freq

# interval_ms = 1000 / fps_animation

# ani = animation.FuncAnimation(
#     fig,
#     update,
#     frames=num_frames,
#     interval=interval_ms,
#     blit=False
# )

# plt.show()