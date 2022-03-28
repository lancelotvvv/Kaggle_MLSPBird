import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd

class LoadAudioDataService:
    def __init__(self, path):
        self.path = path

    def load(self, **args):
        self.y, self.sr = librosa.load(self.path, **args)
        return self
    
    def trim(self, **args):
        self.y, index = librosa.effects.trim(self.y, **args)
        return self
    
    def display(self, **args):
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(self.y, sr=self.sr, **args)
    
    def play(self):
        ipd.Audio(self.y, rate=self.sr)


if __name__ == "__main__":
    audio = LoadAudioDataService('feature_extraction/data/PC1_20090513_050000_0010.wav')
    audio.load().trim().display()

    


