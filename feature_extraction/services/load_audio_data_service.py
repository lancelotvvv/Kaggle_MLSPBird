from ast import Load
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np

class LoadAudioDataService:
    dB = None
    stft = None

    def load(self, path, **args):
        self.path = path
        self.y, self.sr = librosa.load(self.path, **args)
        return self
    
    def trim(self, **args):
        self.y, index = librosa.effects.trim(self.y, **args)
        return self
    
    def display(self, type = 'wave',**args):
        if type == 'wave':
            librosa.display.waveshow(self.y, sr=self.sr, **args)
        elif type == 'spec':
            if self.dB is None:
                self.to_db()
            librosa.display.specshow(self.dB, sr=self.sr, **args)
        else:
            librosa.display.waveshow(self.y, sr=self.sr, **args)

    def play(self):
        return ipd.Audio(self.y, rate=self.sr)

    def to_stft(self):
        self.stft = librosa.stft(self.y)
    
    def to_db(self):
        if self.stft is None:
            self.to_stft()
        self.dB = librosa.amplitude_to_db(np.abs(self.stft))
    
    def hpss_filter(self, **args):
        if self.stft is None:
            self.to_stft()
        H, P = librosa.decompose.hpss(self.stft)
        
        h = librosa.istft(H)
        p = librosa.istft(P)

        HAudio = LoadAudioDataService()
        HAudio.sr = self.sr
        HAudio.stft = H
        HAudio.y = h

        PAudio = LoadAudioDataService()
        PAudio.sr = self.sr
        PAudio.stft = P
        PAudio.y = p

        return HAudio, PAudio



if __name__ == "__main__":
    audio = LoadAudioDataService('feature_extraction/data/PC1_20090513_050000_0010.wav')
    audio.load().trim().display()

    


