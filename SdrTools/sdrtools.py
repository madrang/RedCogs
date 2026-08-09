import asyncio
import io

import discord
from redbot.core import commands

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from rtlsdr import RtlSdr
except ImportError:
    plt = None
    np = None
    RtlSdr = None

SAMPLE_RATE = 2.4e6     # Hz
FFT_SIZE = 2**12        # bins per spectrum chunk
READ_SIZE = 2**18       # total samples per capture
INIT_DISCARD_SAMPLES = 2048


class SdrTools(commands.Cog):
    """SdrTools Cog - RTL-SDR tools for Discord RedBot."""

    __author__ = "Madrang"
    __version__ = "0.0.1"

    def __init__(self, bot):
        self.bot = bot
        # One SDR device: serialize the captures.
        self.sdr_lock = asyncio.Lock()

    @staticmethod
    def capture_spectrum(freq_mhz: float) -> io.BytesIO:
        """Blocking capture + render. Returns a PNG buffer of the spectrum."""
        sdr = RtlSdr()
        try:
            sdr.sample_rate = SAMPLE_RATE
            sdr.center_freq = freq_mhz * 1e6
            sdr.gain = "auto"
            # Drop the first samples: they carry the tuner settling transient.
            sdr.read_samples(INIT_DISCARD_SAMPLES)
            samples = sdr.read_samples(READ_SIZE)
        finally:
            sdr.close()
        # Average log-power spectrum over FFT_SIZE chunks.
        chunks = samples[: len(samples) // FFT_SIZE * FFT_SIZE].reshape(-1, FFT_SIZE)
        window = np.hanning(FFT_SIZE)
        fft_data = np.fft.fftshift(np.fft.fft(chunks * window, axis=1), axes=1)
        power = np.mean(np.abs(fft_data) ** 2, axis=0)
        spectrum = 10 * np.log10(power + 1e-12)
        freqs = np.linspace(
            freq_mhz - SAMPLE_RATE / 2e6
            , freq_mhz + SAMPLE_RATE / 2e6
            , FFT_SIZE
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(freqs, spectrum)
        ax.set_title(f"RF spectrum around {freq_mhz:.3f} MHz")
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Power [dB]")
        ax.grid(True, alpha=0.3)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        return buffer

    @commands.command(name="spectrum")
    async def spectrum(self, ctx: commands.Context, freq_mhz: float) -> None:
        """Capture the RF spectrum around a frequency in MHz and post a picture."""
        if RtlSdr is None or plt is None:
            await ctx.send(
                "The `pyrtlsdr`, `numpy`, or `matplotlib` package is missing. "
                "Reinstall the cog so its requirements are installed."
            )
            return
        if self.sdr_lock.locked():
            await ctx.send("A capture is already running. Wait for it to finish.")
            return
        async with ctx.channel.typing(), self.sdr_lock:
            try:
                buffer = await asyncio.to_thread(self.capture_spectrum, freq_mhz)
            except Exception as e:
                await ctx.send(f"The capture failed: {e}. Is an RTL-SDR device plugged in?")
                return
        await ctx.send(file=discord.File(buffer, filename=f"spectrum_{freq_mhz:.3f}MHz.png"))
