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

from moviepy.editor import VideoFileClip
import tqdm
import os
import numpy as np
from glob import glob
from multiprocessing import Pool
import h5py as h5
from PIL import Image
import cv2
import io

mp4_directory = input('mp4 directory:')
hdf5_directory = input('hdf5 directory:')

n_items = 10000

os.makedirs(hdf5_directory, exist_ok=True)


def extract_utterance(utterance):
    utterance_hdf5_file = hdf5_directory + os.path.sep + utterance[len(mp4_directory):] + '.h5'

    if os.path.exists(utterance_hdf5_file):
        return True

    os.makedirs(os.path.dirname(utterance_hdf5_file), exist_ok=True)

    fps = 16

    try:
        with VideoFileClip(utterance) as video:
            nameformat = '%08d'
            with h5.File(utterance_hdf5_file, 'w') as f:
                tt = np.arange(0, video.duration, 1.0 / fps)
                for i, t in enumerate(tt):
                    name = nameformat % i
                    frame = video.get_frame(t)

                    frame_data = cv2.imencode('.jpg', frame)[1]
                    f.create_dataset(name, data=frame_data)

            video.reader.close()
        del video
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return False
    return True

errors = 0
pool = Pool(1)
for person_dir in tqdm.tqdm(sorted(list(glob(os.path.join(mp4_directory, '*')))), position=0):
    for video_dir in tqdm.tqdm(sorted(list(glob(os.path.join(person_dir, '*')))), position=1, leave=False):
        utterances = sorted(list(glob(os.path.join(video_dir, '*.mp4'))))
        outputs = pool.map(extract_utterance, utterances)
        errors += np.sum(outputs)
            
print(f'{errors} errors occured')
