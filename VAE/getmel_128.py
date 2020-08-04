import librosa
import numpy as np
import glob
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
from PIL import Image

# normalizer-resample的时候重采样到6000
mel_basis = librosa.filters.mel(sr=6000, n_fft=1200, n_mels=128)
def get_mel(y):
    y = librosa.core.stft(y=y, n_fft=1200,
                          hop_length=130,
                          win_length=400,
                          window='hann')
    magnitudes = np.abs(y) ** 2
    mel = np.log10(np.dot(mel_basis, magnitudes) + 1e-6)
    return mel

def extractfeature(file_path, save_path):
    wav, _ = librosa.load(file_path, sr=None)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    mel = get_mel(wav)
    base_name = os.path.basename(file_path)
    base_name = base_name.replace('.wav', '.png')
    save_path = os.path.join(save_path, base_name)
    plt.imsave(save_path, mel)

if __name__ == '__main__':
    PATH = '/home/dengchangxing/HeartLungSound/data/VAE_128/*norm.wav'
    SAVE = '/home/dengchangxing/HeartLungSound/data/VAE_128/'
    if not os.path.exists(SAVE):
        os.makedirs(SAVE)
    files = glob.glob(PATH)
    p = Pool(processes=20)
    for f in files:
        p.apply_async(extractfeature, args=(f,SAVE))
    p.close()
    p.join()