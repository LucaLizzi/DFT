from numba import njit
import numpy as np
from matplotlib import pyplot as plt

SEARCH_R = False

# @jit(nopython=True, fastmath=True)
def rsdft(input_data:None, N:10, sample_rate:int=1, quant_en:bool=True, num_bits:int=8, max_amplitude:float=1.0, 
          r_ideal:float=1.0, r_quant:float=1.0):

    input_data = input_data[:N]
    input_size = input_data.size

    k_vec = np.arange(N)
    arg = 2 * np.pi * k_vec / N

    # check of the constant
    v = [1.0]
    if SEARCH_R and (r_quant in v or r_ideal in v):
        ok = False
        while(ok==False):
            if quant_en:
                twiddles_quant = quantizer(r_quant * np.exp(-1j * arg), num_bits=num_bits)
                if np.all(np.abs(twiddles_quant) <= 1.0): ok = True
                else: r_quant -= 0.001
            else:
                twiddles_ideal = r_ideal * np.exp(-1j * arg)
                if np.all(np.abs(twiddles_ideal) <= 1.0): ok = True
                else: r_ideal -= 0.001            
    else:
        if quant_en:    twiddles_quant = quantizer(r_quant * np.exp(-1j * arg), num_bits=num_bits)
        else:           twiddles_ideal = r_ideal * np.exp(-1j * arg)

    # variables
    out = np.zeros(input_size, dtype=np.float64)
    Z = np.zeros(N, dtype=np.complex128)
    prev_out = 0.0
    err = 0.0

    for index in range(input_size):
        if quant_en:
            err = quantizer(input_data[index] - prev_out, num_bits=num_bits)
            Z = quantizer(quantizer((err + Z), num_bits=num_bits) * twiddles_quant, num_bits=num_bits)
            prev_out = quantizer(np.sum(np.real(Z)) / N, num_bits=num_bits)
        else:
            err = input_data[index] - prev_out
            Z = (err + Z) * twiddles_ideal
            prev_out = np.sum(np.real(Z)) / N

        out[index] = prev_out

    return Z, r_ideal, r_quant
            
# @njit(cache=True)
def rsdft_dc(input_data:None, N:10, sample_rate:int=1, quant_en:bool=True, num_bits:int=8, max_amplitude:float=1.0, 
          r_ideal:float=1.0, r_quant:float=1.0):

    input_data = input_data[:N]
    input_size = input_data.size

    k_vec = np.arange(N)
    arg = 2 * np.pi * k_vec / N

    # check of the constant
    v = [1.0]
    if SEARCH_R and (r_quant in v or r_ideal in v):
        ok = False
        while(ok==False):
            if quant_en:
                twiddles_quant = quantizer(r_quant * np.exp(-1j * arg), num_bits=num_bits)
                if np.all(np.abs(twiddles_quant) <= 1.0): ok = True
                else: r_quant -= 0.001
            else:
                twiddles_ideal = r_ideal * np.exp(-1j * arg)
                if np.all(np.abs(twiddles_ideal) <= 1.0): ok = True
                else: r_ideal -= 0.001            
    else:
        if quant_en:    twiddles_quant = quantizer(r_quant * np.exp(-1j * arg), num_bits=num_bits)
        else:           twiddles_ideal = r_ideal * np.exp(-1j * arg)

    # variables
    out = np.zeros(input_size, dtype=np.float64)
    Z = np.zeros(N, dtype=np.complex128)
    prev_out = 0.0
    err = 0.0

    for index in range(input_size):
        if quant_en:
            err = quantizer(input_data[index] - prev_out, num_bits=num_bits)
            Z = quantizer(quantizer((err + Z), num_bits=num_bits) * twiddles_quant, num_bits=num_bits)
            prev_out = quantizer(np.sum(np.real(Z[0])) / N, num_bits=num_bits)
        else:
            err = input_data[index] - prev_out
            Z = (err + Z) * twiddles_ideal
            prev_out = np.sum(np.real(Z[0])) / N

        out[index] = prev_out

    return Z, r_ideal, r_quant

# @njit(cache=True)
def msdft(input_data:None, N:10, sample_rate:int=1, quant_en:bool=True, num_bits:int=8, max_amplitude:float=1.0, 
          r_ideal:float=1.0, r_quant:float=1.0):

    input_data = input_data[:N]
    input_size = input_data.size

    k_vec = np.arange(N)
    arg = 2 * np.pi * k_vec / N

    # check of the constant
    v = [1.0]
    if SEARCH_R and (r_quant in v or r_ideal in v):
        ok = False
        while(ok==False):
            if quant_en:
                for n in range(input_size):
                    twiddles_quant = quantizer(r_quant * np.exp(-1j * arg * n), num_bits=num_bits)
                    if not np.all(np.abs(twiddles_quant) <= 1.0):   
                        r_quant -= 0.001
                        break
                    else:   ok = True
            else:
                for n in range(input_size):
                    twiddles_ideal = r_ideal * np.exp(-1j * arg * n)
                    if not np.all(np.abs(twiddles_ideal) <= 1.0): 
                        r_ideal -= 0.001
                        break
                    else:   ok = True

    # variables
    out = np.zeros(input_size, dtype=np.float64)
    Z = np.zeros(N, dtype=np.complex128)
    prev_out = 0.0
    err = 0.0

    for index in range(input_size):
        if quant_en:
            err = quantizer(input_data[index] - prev_out, num_bits=num_bits)
            twiddle = quantizer(r_quant * np.exp(-1j * arg * index), num_bits=num_bits)
            prev_out = quantizer(np.sum(np.real(Z * np.conjugate(twiddle))) / N, num_bits=num_bits)
            Z = quantizer(quantizer(r_quant * Z, num_bits=num_bits) + 
                    quantizer(err * twiddle, num_bits=num_bits), num_bits=num_bits)
        else:
            err = input_data[index] - prev_out
            twiddle = r_ideal * np.exp(-1j * arg * index)
            prev_out = np.sum(np.real(Z * np.conjugate(twiddle))) / N
            Z = r_ideal * Z + err * twiddle

        out[index] = prev_out

    return Z, r_ideal, r_quant

# @njit(cache=True)
def msdft_dc(input_data:None, N:10, sample_rate:int=1, quant_en:bool=True, num_bits:int=8, max_amplitude:float=1.0, 
          r_ideal:float=1.0, r_quant:float=1.0):

    input_data = input_data[:N]
    input_size = input_data.size

    k_vec = np.arange(N)
    arg = 2 * np.pi * k_vec / N

    # check of the constant
    v = [1.0]
    if SEARCH_R and (r_quant in v or r_ideal in v):
        ok = False
        while(ok==False):
            if quant_en:
                for n in range(input_size):
                    twiddles_quant = quantizer(r_quant * np.exp(-1j * arg * n), num_bits=num_bits)
                    if not np.all(np.abs(twiddles_quant) <= 1.0):   
                        r_quant -= 0.001
                        break
                    else:   ok = True
            else:
                for n in range(input_size):
                    twiddles_ideal = r_ideal * np.exp(-1j * arg * n)
                    if not np.all(np.abs(twiddles_ideal) <= 1.0): 
                        r_ideal -= 0.001
                        break
                    else:   ok = True

    # variables
    out = np.zeros(input_size, dtype=np.float64)
    Z = np.zeros(N, dtype=np.complex128)
    prev_out = 0.0
    err = 0.0

    for index in range(input_size):
        if quant_en:
            err = quantizer(input_data[index] - prev_out, num_bits=num_bits)
            twiddle = quantizer(r_quant * np.exp(-1j * arg * index), num_bits=num_bits)
            prev_out = quantizer(np.sum(np.real(Z[0])) / N, num_bits=num_bits)
            Z = quantizer(quantizer(r_quant * Z, num_bits=num_bits) + 
                    quantizer(err * twiddle, num_bits=num_bits), num_bits=num_bits)
        else:
            err = input_data[index] - prev_out
            twiddle = r_ideal * np.exp(-1j * arg * index)
            prev_out = np.sum(np.real(Z[0])) / N
            Z = r_ideal * Z + err * twiddle

        out[index] = prev_out

    return Z, r_ideal, r_quant


def plt_func(wave_name, wave_data, num_bits, max_amplitude, 
    rsdft, rsdft_dc, msdft, msdft_dc, fft, horiz, max_freq, quant_en,
    r_rsdft_ideal, r_rsdft_quant, r_rsdft_dc_ideal, r_rsdft_dc_quant,
    r_msdft_ideal, r_msdft_quant, r_msdft_dc_ideal, r_msdft_dc_quant):

    x = np.linspace(0, max_freq/1000, len(horiz))

    fig, ax = plt.subplots(2, 3, figsize=(18, 15))
    axes_flat = ax.flatten()

    ax[0,0].plot(x, rsdft, color='gray', alpha=0.6)
    ax[0,0].set_title(f"rsdft (r={r_rsdft_quant if quant_en else r_rsdft_ideal})")
    ax[0,0].set_xlabel("Frequency (kHz)")
    ax[0,0].set_xscale('log')

    ax[0,1].plot(x, rsdft_dc, color='gray', alpha=0.6)
    ax[0,1].set_title(f"rsdft_dc (r={r_rsdft_dc_quant if quant_en else r_rsdft_dc_ideal})")
    ax[0,1].set_xlabel("Frequency (kHz)")
    ax[0,1].set_xscale('log')

    ax[0,2].plot(x, msdft, color='gray', alpha=0.6)
    ax[0,2].set_title(f"msdft (r={r_msdft_quant if quant_en else r_msdft_ideal})")
    ax[0,2].set_xlabel("Frequency (kHz)")
    ax[0,2].set_xscale('log')

    ax[1,1].plot(x, fft, color='gray', alpha=0.6)
    ax[1,1].set_title("fft")
    ax[1,1].set_xlabel("Frequency (kHz)")
    ax[1,1].set_xscale('log')

    ax[1,2].plot(x, msdft_dc, color='gray', alpha=0.6)
    ax[1,2].set_title(f"msdft_dc (r={r_msdft_dc_quant if quant_en else r_msdft_dc_ideal})")
    ax[1,2].set_xlabel("Frequency (kHz)")
    ax[1,2].set_xscale('log')

    signal = wave_data - np.mean(wave_data)
    zero_crossings = np.where((signal[:-1] <= 0) & (signal[1:] > 0))[0]
    if len(zero_crossings) >= 2:
            period_samples = zero_crossings[1] - zero_crossings[0]
    else:
        period_samples = len(wave_data)
    num_periods_to_show = 1
    ax[1,0].plot(range(len(wave_data)), wave_data, color='gray', alpha=0.6)
    ax[1,0].set_title(f"wave data ({num_periods_to_show} periods)")
    ax[1,0].set_xlim(0, period_samples * num_periods_to_show)
    ax[1,0].set_ylabel("Amplitude")
    ax[1,0].set_xlabel(f"Time (periods: {num_periods_to_show})")

    if "quant" in wave_name:
        fig.suptitle(f"wave name: {wave_name}, num_bits: {num_bits}")#, fontsize=14)
    if "no_quant" in wave_name:
        fig.suptitle(f"wave name: {wave_name}")#, fontsize=14)
    fig.tight_layout()

    plt.savefig(f"./images/{wave_name}.png", dpi=300, bbox_inches='tight')

    plt.close()


# @njit(cache=True)
def quantizer(signal, num_bits:int=8, max_amplitude:float=1.0):
    if num_bits == "no quant":
        raise Exception("Quantization is disabled. Please provide a valid number of bits.")

    def q(s):
        half_levels = 2 ** (num_bits - 1)
        max_bits = half_levels      # max 128 with 8 bits
        min_bits = -half_levels + 1 # min -127 with 8 bits
        
        num = max_amplitude * 2
        denum = half_levels * 2
        resolution = num / denum

        nums = np.round(s / resolution)
        # nums = np.floor(s / resolution)       # wrong idea
        return nums * resolution

    if np.iscomplexobj(signal):
        return q(np.real(signal)) + 1j * q(np.imag(signal))
    else:
        return q(signal)


def quantizer_ext(signal, num_bits:int=8, max_amplitude:float=1.0):
    if num_bits == "no quant":
        raise Exception("Quantization is disabled. Please provide a valid number of bits.")

    def q(s):
        half_levels = 2 ** (num_bits - 1)
        max_bits = half_levels      # max 128 with 8 bits
        min_bits = -half_levels + 1 # min -127 with 8 bits
        
        num = max_amplitude * 2
        denum = half_levels * 2
        resolution = num / denum

        nums = np.round(s / resolution)
        # nums = np.floor(s / resolution)       # wrong idea
        return nums * resolution

    if np.iscomplexobj(signal):
        return q(np.real(signal)) + 1j * q(np.imag(signal))
    else:
        return q(signal)
