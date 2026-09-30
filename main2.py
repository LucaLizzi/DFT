from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from numba import jit
import matplotlib.animation as animation


# ██╗   ██╗ █████╗ ██████╗ ██╗ █████╗ ██████╗ ██╗     ███████╗███████╗
# ██║   ██║██╔══██╗██╔══██╗██║██╔══██╗██╔══██╗██║     ██╔════╝██╔════╝
# ██║   ██║███████║██████╔╝██║███████║██████╔╝██║     █████╗  ███████╗
# ╚██╗ ██╔╝██╔══██║██╔══██╗██║██╔══██║██╔══██╗██║     ██╔══╝  ╚════██║
#  ╚████╔╝ ██║  ██║██║  ██║██║██║  ██║██████╔╝███████╗███████╗███████║
#   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝╚══════╝
                                                                    
sample_rate, data = wavfile.read("test3_notes.wav")

# uncomment for test
# sample_rate = 44100
# freq1 = 200
# t = np.linspace(0, 1, sample_rate)
# data = np.sin(2 * np.pi * freq1 * t)

window_size = 4096
fps_animation = 30


# ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
# ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
# █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
# ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
# ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
# ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
                                                                          
@jit(nopython=True, fastmath=True)
def _sdft_kernel(input_data, N, fps_animation, sample_rate):
    input_size = input_data.size

    # number of samples to wait before save the spectrum (for animation)
    idle_samples = sample_rate // fps_animation
    animation_list = np.zeros((int(input_size / idle_samples), N), dtype=np.complex128)
    total_frames = int(input_size / idle_samples)
    
    r = 1       # r<1 useful in case of divergence
    k_vec = np.arange(N)
    arg = 2 * np.pi * k_vec / N
    twiddles = r * (np.cos(arg) - 1j * np.sin(arg))

    output = np.zeros(input_size, dtype=np.float64)
    Z = np.zeros(N, dtype=np.complex128)

    prev_out = 0.0
    anim_index = 0

    for index in range(input_size):
        err = input_data[index] - prev_out

        Z = (err + Z) * twiddles

        prev_out = np.real(np.sum(Z)) / N
        # prev_out = np.real(Z[0]) / N      # needs the convergence factor r<1
        output[index] = prev_out

        if (index + 1) % idle_samples == 0 and anim_index < total_frames:
            animation_list[anim_index] = Z.copy()    # last column
            anim_index += 1

    return output, 0, animation_list

# kernel warmp up, the first call is slow (from Gemini) 
_ = _sdft_kernel(np.zeros(10, dtype=np.float64), 8, 30.0, 44100.0)



# ███████╗██████╗ ███████╗ ██████╗████████╗██████╗ ██╗   ██╗███╗   ███╗     █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗██╗███████╗
# ██╔════╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔══██╗██║   ██║████╗ ████║    ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝██╔════╝██║██╔════╝
# ███████╗██████╔╝█████╗  ██║        ██║   ██████╔╝██║   ██║██╔████╔██║    ███████║██╔██╗ ██║███████║██║   ╚████╔╝ ███████╗██║███████╗
# ╚════██║██╔═══╝ ██╔══╝  ██║        ██║   ██╔══██╗██║   ██║██║╚██╔╝██║    ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝  ╚════██║██║╚════██║
# ███████║██║     ███████╗╚██████╗   ██║   ██║  ██║╚██████╔╝██║ ╚═╝ ██║    ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████║██║███████║
# ╚══════╝╚═╝     ╚══════╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚═╝╚══════╝
                                                                                                                                                

# normalization step
if data.dtype == np.int16:
    data = data.astype(np.float64) / 32768.0
elif data.dtype == np.int32:
    data = data.astype(np.float64) / 2147483648.0
elif data.dtype == np.uint8:
    data = (data.astype(np.float64) - 128.0) / 128.0
else:
    data = data.astype(np.float64)

audio_seconds = data.size/sample_rate

print(f"Sample rate input file: {sample_rate} (Hz)")
print(f"Input data file duration: {audio_seconds} (s)")
print(f"Input data file num. samples: {data.size}")
print(f"Spectrum calculation started")

output, Xkn, spectr_fps = _sdft_kernel(data, window_size, fps_animation, sample_rate)

print("Finished")



#  █████╗ ███╗   ██╗██╗███╗   ███╗ █████╗ ████████╗███████╗██████╗     ██████╗ ██╗      ██████╗ ████████╗
# ██╔══██╗████╗  ██║██║████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██╔══██╗    ██╔══██╗██║     ██╔═══██╗╚══██╔══╝
# ███████║██╔██╗ ██║██║██╔████╔██║███████║   ██║   █████╗  ██║  ██║    ██████╔╝██║     ██║   ██║   ██║   
# ██╔══██║██║╚██╗██║██║██║╚██╔╝██║██╔══██║   ██║   ██╔══╝  ██║  ██║    ██╔═══╝ ██║     ██║   ██║   ██║   
# ██║  ██║██║ ╚████║██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗██████╔╝    ██║     ███████╗╚██████╔╝   ██║   
# ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═════╝     ╚═╝     ╚══════╝ ╚═════╝    ╚═╝   
                                                                                                       
# (help from Gemini)
# avoid negative spectrum
half_N = window_size // 2
freq_axis = np.linspace(0, sample_rate / 2, half_N)

# takes in account the first half_N samples of the spectrum
spectr_mag = np.abs(spectr_fps[:, :half_N]) / (window_size / 2)
num_frames = spectr_mag.shape[0]
spectr_db = 20 * np.log10(spectr_mag + 1e-12)   #1e-12 needed to avoid inf values


fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 7))

# --- 1. TIME PLOT ---
x_time = np.linspace(0, audio_seconds, data.size)
ax_time.plot(x_time, data, color='gray', alpha=0.6)
line_cursor = ax_time.axvline(x=0, color='red', linestyle='--')

ax_time.set_title("Time signal")
ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("Amplitude")
ax_time.set_xlim(0, audio_seconds)
ax_time.grid(True)

# --- 2. FREQUENCY PLOT ---
(line_freq,) = ax_freq.plot(freq_axis, np.full(half_N, -120.0), color="crimson", lw=1.5)

ax_freq.set_title("Spectrum SDFT")
ax_freq.set_xlabel("Frequency (Hz)")
ax_freq.set_ylabel("Amplitude (dB)")
ax_freq.grid(True)
# ax_freq.set_xlim(0, 2000)  # uncomment to zoom in

ax_freq.set_xscale('log')

max_db = np.max(spectr_db)
ax_freq.set_ylim(-120, max(max_db + 5, 0))
ax_freq.grid(True)

plt.tight_layout()

# --- 3. ANIMATION ---
def update(i):
    time_now = i * (audio_seconds / num_frames)

    line_cursor.set_xdata([time_now])

    line_freq.set_ydata(spectr_db[i])

    ax_freq.set_title(f"Spettro SDFT a t = {time_now:.2f} s")

    return line_cursor, line_freq

interval_ms = 1000 / fps_animation

ani = animation.FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=interval_ms,
    blit=False
)

plt.show()