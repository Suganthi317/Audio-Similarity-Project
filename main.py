import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
audio1_path = "audio/audio1.mp3"
audio2_path = "audio/audio1.mp3"

sr = 16000          
duration_sec = 19   

y1, _ = librosa.load(audio1_path, sr=sr, mono=True, duration=duration_sec)
y2, _ = librosa.load(audio2_path, sr=sr, mono=True, duration=duration_sec)

y1, _ = librosa.effects.trim(y1, top_db=25)
y2, _ = librosa.effects.trim(y2, top_db=25)

y1 = y1 / (np.max(np.abs(y1)) + 1e-9)
y2 = y2 / (np.max(np.abs(y2)) + 1e-9)

min_len = min(len(y1), len(y2))
y1 = y1[:min_len]
y2 = y2[:min_len]

fft1 = np.fft.rfft(y1)
fft2 = np.fft.rfft(y2)
mag1 = np.abs(fft1)
mag2 = np.abs(fft2)
freqs = np.fft.rfftfreq(len(y1), d=1/sr)

mag1_db = librosa.amplitude_to_db(mag1, ref=np.max)
mag2_db = librosa.amplitude_to_db(mag2, ref=np.max)

plt.figure(figsize=(12, 6))
plt.plot(freqs, mag1_db, label="Audio 1")
plt.plot(freqs, mag2_db, label="Audio 2")
plt.title("Frequency Domain Comparison (FFT Spectrum)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("output/frequency_comparison.png")
plt.show()
print("Frequency graph saved: output/frequency_comparison.png")

mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13)
mfcc1_mean = np.mean(mfcc1, axis=1)
mfcc2_mean = np.mean(mfcc2, axis=1)
mfcc_similarity = 1 - cosine(mfcc1_mean, mfcc2_mean)
mfcc_similarity = np.clip(mfcc_similarity, 0, 1)

centroid1 = librosa.feature.spectral_centroid(y=y1, sr=sr)[0]
centroid2 = librosa.feature.spectral_centroid(y=y2, sr=sr)[0]
centroid1_mean = np.mean(centroid1)
centroid2_mean = np.mean(centroid2)
centroid_similarity = 1 - (abs(centroid1_mean - centroid2_mean) / (max(centroid1_mean, centroid2_mean) + 1e-9))
centroid_similarity = np.clip(centroid_similarity, 0, 1)

rms1 = librosa.feature.rms(y=y1)[0]
rms2 = librosa.feature.rms(y=y2)[0]
rms1_mean = np.mean(rms1)
rms2_mean = np.mean(rms2)
rms_similarity = 1 - (abs(rms1_mean - rms2_mean) / (max(rms1_mean, rms2_mean) + 1e-9))
rms_similarity = np.clip(rms_similarity, 0, 1)

final_similarity = (0.5 * mfcc_similarity) + (0.3 * centroid_similarity) + (0.2 * rms_similarity)
final_similarity_percent = final_similarity * 100

print("\n--- Feature Summary ---")
print("MFCC Mean Vector (Audio 1):", np.round(mfcc1_mean, 2))
print("MFCC Mean Vector (Audio 2):", np.round(mfcc2_mean, 2))

print("\nSpectral Centroid Mean (Audio 1):", round(centroid1_mean, 2))
print("Spectral Centroid Mean (Audio 2):", round(centroid2_mean, 2))

print("\nRMS Energy Mean (Audio 1):", round(rms1_mean, 4))
print("RMS Energy Mean (Audio 2):", round(rms2_mean, 4))

print("\n--- Similarity Results ---")
print(f"MFCC Similarity           : {mfcc_similarity:.4f}")
print(f"Spectral Centroid Similar : {centroid_similarity:.4f}")
print(f"RMS Energy Similarity     : {rms_similarity:.4f}")

print(f"\nFINAL Similarity Score  : {final_similarity_percent:.2f}%")

if final_similarity_percent >= 80:
    conclusion = "Very Similar"
elif final_similarity_percent >= 60:
    conclusion = "Moderately Similar"
elif final_similarity_percent >= 40:
    conclusion = "Slightly Similar"
else:
    conclusion = "Not Similar"

print("Conclusion:", conclusion)
