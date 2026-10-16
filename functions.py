from numba import jit
import numpy as np
from matplotlib import pyplot as plt

# @jit(nopython=True, fastmath=True)
def _sdft_kernel(input_data:None, N:10, fps_animation:int=30, sample_rate:int=1, params:tuple=(False, False, False, False), animation:bool=False):
    r_sdft, dc_sdft, m_sdft, fft = params
    out_rsdft, out_rsdft_dc, out_msdft, out_fft = None, None, None, None
    if input_data.all() != None:
        if fft:
            fft_res = np.fft.fft(input_data[:N])
            freqs = np.fft.fftfreq(len(input_data), 1/sample_rate)
            out_fft = np.abs(fft_res)
            spectr_fft_dB = 20 * np.log10(out_fft + 1e-12)   #1e-12 needed to avoid inf values

        if r_sdft or m_sdft or dc_sdft:
            input_data = input_data[:N]
            input_size = input_data.size

            # number of samples to wait before save the spectrum (for animation)
            if animation:   
                idle_samples = sample_rate // fps_animation
                animation_list = np.zeros((int(input_size / idle_samples), N), dtype=np.complex128)
                total_frames = int(input_size / idle_samples)
                anim_index = 0

            r = 1
            k_vec = np.arange(N)
            arg = 2 * np.pi * k_vec / N
            if r_sdft:
                if dc_sdft: r = 0.999
                twiddles = r * (np.cos(arg) - 1j * np.sin(arg))
            if m_sdft:
                r = 0.999

            out_rsdft = np.zeros(input_size, dtype=np.float64)
            out_rsdft_dc = np.zeros(input_size, dtype=np.float64)
            out_msdft = np.zeros(input_size, dtype=np.float64)
            
            Z_rsdft = np.zeros(N, dtype=np.complex128)
            Z_rsdft_dc = np.zeros(N, dtype=np.complex128)
            Z_msdft = np.zeros(N, dtype=np.complex128)

            prev_out_rsdft = 0.0
            prev_out_msdft = 0.0
            prev_out_rsdft_dc = 0.0

            spectr_rsdft_dB = np.zeros(N, dtype=np.complex128)
            spectr_msdft_dB = np.zeros(N, dtype=np.complex128)
            spectr_rsdft_DC_dB = np.zeros(N, dtype=np.complex128)
            spectr_fft_dB = np.zeros(N, dtype=np.complex128)

            err_rsdft = 0.0
            err_rsdft_dc = 0.0
            err_msdft = 0.0

            for index in range(input_size):
                # err = input_data[index] - prev_out

                if r_sdft:      # resonator branch
                    err_rsdft = input_data[index] - prev_out_rsdft
                    Z_rsdft = (err_rsdft + Z_rsdft) * twiddles
                    prev_out_rsdft = np.sum(np.real(Z_rsdft)) / N
                    out_rsdft[index] = prev_out_rsdft

                if dc_sdft:       # DC components with resonantor branch
                    err_rsdf_dc = input_data[index] - prev_out_rsdft_dc
                    Z_rsdft_dc = (err_rsdf_dc + Z_rsdft_dc) * twiddles
                    prev_out_rsdft_dc = np.real(Z_rsdft_dc[0]) / N      # faster but needs the convergence factor r<1
                    out_rsdft_dc[index] = prev_out_rsdft_dc

                if m_sdft:         # modulator branch
                    err_msdft = input_data[index] - prev_out_msdft
                    twiddle = np.exp(1j * arg * index)
                    prev_out_msdft = np.sum(np.real(Z_msdft * np.conjugate(twiddle))) / N
                    Z_msdft = r * Z_msdft + err_msdft * twiddle
                    out_msdft[index] = prev_out_msdft

                # output[index] = prev_out

                if animation:
                    if (index + 1) % idle_samples == 0 and anim_index < total_frames:
                        animation_list[anim_index] = Z_rsdft.copy()    # last column
                        anim_index += 1

            if animation: return None, 0, animation_list
        
        return Z_rsdft, Z_rsdft_dc, Z_msdft, out_fft
        # return out_rsdft, out_rsdft_dc, out_msdft, out_fft
        # return spectr_rsdft_dB, spectr_msdft_dB, spectr_rsdft_DC_dB, spectr_fft_dB
    else:
        raise Exception("No data")



def plt_func(wave_name, rsdft, err_rsdft, 
    rsdft_dc, err_rsdft_dc, 
    msdft, err_msdft, 
    fft, x):

    fig, ax = plt.subplots(3, 3, figsize=(12, 10))
    axes_flat = ax.flatten()

    ax[0,0].plot(x, rsdft, color='gray', alpha=0.6)
    ax[0,0].set_title("rsdft")
    ax[0,0].set_xlabel("Frequency Hz")

    ax[1,0].plot(x, err_rsdft, color='gray', alpha=0.6)
    ax[1,0].set_title("err rsdft")
    ax[1,0].set_xlabel("Frequency Hz")

    ax[0,1].plot(x, rsdft_dc, color='gray', alpha=0.6)
    ax[0,1].set_title("rsdft_dc")
    ax[0,1].set_xlabel("Frequency Hz")

    ax[1,1].plot(x, err_rsdft_dc, color='gray', alpha=0.6)
    ax[1,1].set_title("err rsdft_dc")
    ax[1,1].set_xlabel("Frequency Hz")

    ax[0,2].plot(x, msdft, color='gray', alpha=0.6)
    ax[0,2].set_title("msdft")
    ax[0,2].set_xlabel("Frequency Hz")

    ax[1,2].plot(x, err_msdft, color='gray', alpha=0.6)
    ax[1,2].set_title("err msdft")
    ax[1,2].set_xlabel("Frequency Hz")

    ax[2,1].plot(x, fft, color='gray', alpha=0.6)
    ax[2,1].set_title("fft")
    ax[2,1].set_xlabel("Frequency Hz")

    ax[2, 0].axis("off")
    ax[2, 2].axis("off")

    fig.suptitle(f"wave name: {wave_name}")
    fig.tight_layout()

    plt.savefig(f"./images/{wave_name}.png", dpi=300, bbox_inches='tight')

    plt.close()