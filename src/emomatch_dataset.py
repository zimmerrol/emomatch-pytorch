# MIT License
# 
# Copyright (c) 2018 Roland Zimmermann
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import utility as ut
import os
import torch
from glob import glob
from tqdm import tqdm

class EmoMatchDataset(Dataset):
    """EmoMatch dataset using VoxCeleb v2"""

    def __init__(self, directory, map_correct=True, n_video_samples=16, image_transform=None, stft_transform=None, size=None, number_of_videos=10000):
        self._directory = directory
        self._map_correct = map_correct
        self._n_video_samples = n_video_samples
        self._image_transform = image_transform
        self._stft_transform = stft_transform

        if size > 0:
            _h5_video_names = sorted(list(glob(os.path.join(directory, 'h5/**/*'))))[:size]
            _wav_video_names = sorted(list(glob(os.path.join(directory, 'wav/**/*'))))[:size]
        else:
            _h5_video_names = sorted(list(glob(os.path.join(directory, 'h5/**/*'))))[:number_of_videos][size:]
            _wav_video_names = sorted(list(glob(os.path.join(directory, 'wav/**/*'))))[:number_of_videos][size:]

        self._utterance_names = dict()

        for name in tqdm(_h5_video_names):
            if name.replace('/h5/', '/wav/') in _wav_video_names:
                self._utterance_names[name] = [os.path.splitext(os.path.splitext(os.sep.join(os.path.normpath(x).split(os.sep)[-3:]))[0])[0] for x in list(glob(os.path.join(name, '*.mp4.h5')))]

        self._video_names = sorted(list(self._utterance_names.keys()))

        self._cumsum = np.cumsum([len(self._utterance_names[x]) for x in self._video_names])

    def __len__(self):
        return self._cumsum[-1]

    @staticmethod
    def _next_smallest(value, array):
        nearest = np.argmax(array > value)
        return nearest, value - array[nearest]

    @property
    def map_correct(self):
        return self._map_correct

    @map_correct.setter
    def map_correct(self, value):
        self._map_correct = value

    def __getitem__(self, video_idx):
        try:
            audio_idx = video_idx
            if not self._map_correct:
                while EmoMatchDataset._next_smallest(audio_idx, self._cumsum)[0] == EmoMatchDataset._next_smallest(video_idx, self._cumsum)[0]:
                    audio_idx = np.random.randint(self.__len__())

            video_id, video_utterance_id = EmoMatchDataset._next_smallest(video_idx, self._cumsum)
            audio_id, audio_utterance_id = EmoMatchDataset._next_smallest(audio_idx, self._cumsum)

            mp4_video_name = self._video_names[video_id]
            wav_video_name = self._video_names[audio_id]

            video_utterance_name = self._utterance_names[mp4_video_name][video_utterance_id] + '.mp4' + '.h5'
            audio_utterance_name = self._utterance_names[wav_video_name][audio_utterance_id] + '.wav'

            # create audio sample
            try:
                wav_data = ut.load_audio_sample(os.path.join(self._directory, 'wav', audio_utterance_name))
                wav_data = ut.create_audio_sample(wav_data)
            except:
                print('Error in:', os.path.join(self._directory, 'wav', audio_utterance_name))
                return None

            audio_stft = ut.extract_spectrum(wav_data)
            audio_stft = np.abs(audio_stft)
            audio_stft = np.log(audio_stft + 1e-10)

            if self._stft_transform:
                audio_stft = self._stft_transform(audio_stft)

            audio_stft = audio_stft.reshape((1, *audio_stft.shape))

            try:
                video_features = ut.create_video_sample_hdf5(os.path.join(self._directory, 'h5', video_utterance_name), self._n_video_samples)
            except Exception as ex:
                import traceback
                print('Error in:', os.path.join(self._directory, 'h5', video_utterance_name), ex)
                traceback.print_exc()
                return None

            try:
                if self._image_transform:
                    video_features = self._image_transform(video_features)
            except Exception as ex:
                import traceback
                print('Error in processing:', os.path.join(self._directory, 'mp4', video_utterance_name), ex)
                traceback.print_exc()
                return None

            audio_stft = torch.from_numpy(audio_stft.astype(dtype=np.float32))
            video_features = torch.from_numpy(video_features.astype(dtype=np.float32))
            video_features = video_features.permute(0, 3, 1, 2)

            targets = torch.zeros((1,), dtype=torch.long)
            targets[0] = int(self._map_correct)

            return ((audio_stft, video_features), targets)
        except Exception as ex:
            import traceback
            print('Error __getitem__:', ex)
            traceback.print_exc()
            return None
