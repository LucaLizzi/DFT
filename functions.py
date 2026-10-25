from numba import jit
import numpy as np
from matplotlib import pyplot as plt

# @jit(nopython=True, fastmath=True)
def _sdft_kernel(input_data:None, N:10, fps_animation:int=30, 
    sample_rate:int=1, params:tuple=(False, False, False, False), 
    animation:bool=False, quant_en:bool=True, num_bits:int=8, max_amplitude:float=100.0):
    
    r_sdft, rsdft_dc, m_sdft, msdft_dc, fft = params
    out_rsdft, out_rsdft_dc, out_msdft, out_fft = None, None, None, None

    if input_data.all() != None:
        if fft:
            out_fft = np.fft.fft(input_data[:N])
            # freqs = np.fft.fftfreq(len(input_data), 1/sample_rate)
            # out_fft = np.abs(fft_res)
            # spectr_fft_dB = 20 * np.log10(out_fft + 1e-12)   #1e-12 needed to avoid inf values

        if r_sdft or m_sdft or rsdft_dc or msdft_dc:
            input_data = input_data[:N]
            input_size = input_data.size

            # number of samples to wait before save the spectrum (for animation)
            if animation:   
                idle_samples = sample_rate // fps_animation
                animation_list = np.zeros((int(input_size / idle_samples), N), dtype=np.complex128)
                total_frames = int(input_size / idle_samples)
                anim_index = 0

            k_vec = np.arange(N)
            arg = 2 * np.pi * k_vec / N
            if r_sdft:
                r_rsdft = 0.99
                if quant_en:
                    twiddles = quantizer(r_rsdft * np.cos(arg), num_bits=num_bits, max_amplitude=r_rsdft) \
                                + 1j * quantizer(r_rsdft * np.sin(arg), num_bits=num_bits, max_amplitude=r_rsdft)
                else:
                    twiddles = r_rsdft * np.cos(arg) + 1j * r_rsdft * np.sin(arg)
            if rsdft_dc:
                r_rsdft_dc = 0.99
                if quant_en:
                    twiddles_dc = quantizer(r_rsdft_dc * np.cos(arg), num_bits=num_bits, max_amplitude=r_rsdft_dc) \
                                + 1j * quantizer(r_rsdft_dc * np.sin(arg), num_bits=num_bits, max_amplitude=r_rsdft_dc)
                else:
                    twiddles_dc = r_rsdft_dc * np.cos(arg) + 1j * r_rsdft_dc * np.sin(arg)
            if m_sdft:
                r_msdft = 1#0.999
            if msdft_dc:
                r_msdft_dc = 1#0.999

            out_rsdft = np.zeros(input_size, dtype=np.float64)
            out_rsdft_dc = np.zeros(input_size, dtype=np.float64)
            out_msdft = np.zeros(input_size, dtype=np.float64)
            out_msdft_dc = np.zeros(input_size, dtype=np.float64)
            
            Z_rsdft = np.zeros(N, dtype=np.complex128)
            Z_rsdft_dc = np.zeros(N, dtype=np.complex128)
            Z_msdft = np.zeros(N, dtype=np.complex128)
            Z_msdft_dc = np.zeros(N, dtype=np.complex128)

            prev_out_rsdft = 0.0
            prev_out_msdft = 0.0
            prev_out_rsdft_dc = 0.0
            prev_out_msdft_dc = 0.0

            err_rsdft = 0.0
            err_rsdft_dc = 0.0
            err_msdft = 0.0
            err_msdft_dc = 0.0

            for index in range(input_size):

                if r_sdft:      # resonator branch
                    if quant_en:
                        err_rsdft = quantizer(input_data[index] - prev_out_rsdft, num_bits=num_bits)
                        Z_rsdft = quantizer(quantizer((err_rsdft + Z_rsdft), num_bits=num_bits) * twiddles, num_bits=num_bits)
                        prev_out_rsdft = quantizer(np.sum(np.real(Z_rsdft)) / N, num_bits=num_bits)
                    else:
                        err_rsdft = input_data[index] - prev_out_rsdft
                        Z_rsdft = (err_rsdft + Z_rsdft) * twiddles
                        prev_out_rsdft = np.sum(np.real(Z_rsdft)) / N

                    out_rsdft[index] = prev_out_rsdft

                if rsdft_dc:       # DC components with resonantor branch
                    if quant_en:
                        err_rsdft_dc = quantizer(input_data[index] - prev_out_rsdft_dc, num_bits=num_bits)
                        Z_rsdft_dc = quantizer(quantizer((err_rsdft_dc + Z_rsdft_dc), num_bits=num_bits) * twiddles_dc, num_bits=num_bits)
                        prev_out_rsdft_dc = quantizer(np.real(Z_rsdft_dc[0]) / N, num_bits=num_bits)      # faster but needs the convergence factor r<1
                    else:
                        err_rsdft_dc = input_data[index] - prev_out_rsdft_dc
                        Z_rsdft_dc = (err_rsdft_dc + Z_rsdft_dc) * twiddles_dc
                        prev_out_rsdft_dc = np.real(Z_rsdft_dc[0]) / N      # faster but needs the convergence factor r<1
                    
                    out_rsdft_dc[index] = prev_out_rsdft_dc

                if m_sdft:         # modulator branch
                    if quant_en:
                        err_msdft = quantizer(input_data[index] - prev_out_msdft, num_bits=num_bits)
                        twiddle = quantizer(r_msdft * np.cos(arg * index), num_bits=num_bits, max_amplitude=r_msdft) \
                                    + 1j * quantizer(r_msdft * np.sin(arg * index), num_bits=num_bits, max_amplitude=r_msdft)       # TODO r may be removed with quantization
                        prev_out_msdft = quantizer(np.sum(np.real(Z_msdft * np.conjugate(twiddle))) / N, num_bits=num_bits)
                        Z_msdft = quantizer(quantizer(r_msdft * Z_msdft, num_bits=num_bits) + 
                                quantizer(err_msdft * twiddle, num_bits=num_bits))
                    else:
                        err_msdft = input_data[index] - prev_out_msdft
                        twiddle = r_msdft * np.cos(arg * index) + 1j * r_msdft * np.sin(arg * index)
                        prev_out_msdft = np.sum(np.real(Z_msdft * np.conjugate(twiddle))) / N
                        Z_msdft = r_msdft * Z_msdft + err_msdft * twiddle

                    out_msdft[index] = prev_out_msdft

                if msdft_dc:
                    if quant_en:
                        err_msdft_dc = quantizer(input_data[index] - prev_out_msdft_dc, num_bits=num_bits)
                        twiddle__ = quantizer(r_msdft_dc * np.cos(arg * index), num_bits=num_bits, max_amplitude=r_msdft_dc) \
                                    + 1j * quantizer(r_msdft_dc * np.sin(arg * index), num_bits=num_bits, max_amplitude=r_msdft_dc)       # TODO r may be removed with quantization
                        prev_out_msdft_dc = quantizer(np.sum(np.real(Z_msdft_dc[0] * np.conjugate(twiddle__))) / N, num_bits=num_bits)      # faster but needs the convergence factor r<1
                        Z_msdft_dc = quantizer(quantizer(r_msdft_dc * Z_msdft_dc, num_bits=num_bits) + 
                                quantizer(err_msdft_dc * twiddle__, num_bits=num_bits))
                    else:
                        err_msdft_dc = input_data[index] - prev_out_msdft_dc
                        twiddle_ = r_msdft_dc * np.cos(arg * index) + 1j * r_msdft_dc * np.sin(arg * index)
                        prev_out_msdft_dc = np.sum(np.real(Z_msdft_dc[0] * np.conjugate(twiddle_))) / N      # faster but needs the convergence factor r<1
                        Z_msdft_dc = r_msdft_dc * Z_msdft_dc + err_msdft_dc * twiddle_

                    out_msdft_dc[index] = prev_out_msdft_dc

                if animation:
                    if (index + 1) % idle_samples == 0 and anim_index < total_frames:
                        animation_list[anim_index] = Z_rsdft.copy()    # last column
                        anim_index += 1

            if animation: return None, 0, animation_list
        
        return Z_rsdft, Z_rsdft_dc, Z_msdft, Z_msdft_dc, out_fft

    else:
        raise Exception("No data")


def plt_func(wave_name, wave_data, num_bits, max_amplitude, 
    rsdft, rsdft_dc, msdft, msdft_dc, fft, horiz, max_freq):

    x = np.linspace(0, max_freq/1000, len(horiz))

    fig, ax = plt.subplots(2, 3, figsize=(18, 15))
    axes_flat = ax.flatten()

    ax[0,0].plot(x, rsdft, color='gray', alpha=0.6)
    ax[0,0].set_title("rsdft")
    ax[0,0].set_xlabel("Frequency (kHz)")
    ax[0,0].set_xscale('log')

    ax[0,1].plot(x, rsdft_dc, color='gray', alpha=0.6)
    ax[0,1].set_title("rsdft_dc")
    ax[0,1].set_xlabel("Frequency (kHz)")
    ax[0,1].set_xscale('log')

    ax[0,2].plot(x, msdft, color='gray', alpha=0.6)
    ax[0,2].set_title("msdft")
    ax[0,2].set_xlabel("Frequency (kHz)")
    ax[0,2].set_xscale('log')

    ax[1,1].plot(x, fft, color='gray', alpha=0.6)
    ax[1,1].set_title("fft")
    ax[1,1].set_xlabel("Frequency (kHz)")
    ax[1,1].set_xscale('log')

    ax[1,2].plot(x, msdft_dc, color='gray', alpha=0.6)
    ax[1,2].set_title("msdft_dc")
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