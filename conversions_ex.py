from pydub import AudioSegment
import numpy as np
from python_speech_features import mfcc


def convert_to_wav(inp, out):
    sound = AudioSegment.from_mp3(inp)
    sound.export(out, format="wav")


import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from python_speech_features.sigproc import framesig, magspec

(rate, sig) = wav.read("data/jff.wav")
print(rate)
# get [10s, 25s] interval
dt = sig[10 * rate: 25 * rate]
wav.write("data/res/extract.wav", rate, dt)
print(f"Sample count {len(dt)}")

frame_size = 0.025 * rate
frame_step = 0.010 * rate

print(f"Frame size {frame_size} and frame step {frame_step}")

framed_sig = framesig(dt, frame_size, frame_step, winfunc=np.hamming)
print(f"Framed signal: {framed_sig.shape}")

spectrum = magspec(framed_sig, frame_size)
spectrum_2 = np.abs(np.fft.fft(framed_sig))
frequencies = np.fft.fftfreq(int(frame_size)) * frame_size * ( 1 / 0.025)

print(f"Frequencies {frequencies} with step size: {rate / frame_size}Hz")
print(f"Spectrum after FFT: {spectrum.shape}")

# plt.plot(range(0, len(dt)), dt)
plt.imshow(np.flipud(spectrum_2.T))
# plt.show()

res = mfcc(sig, rate)
print("Done")

# convert_to_wav("data/raw/jff.mp3", "data/jff.wav")
