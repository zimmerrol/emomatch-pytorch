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

import numpy as np
from glob import glob
import os
from subprocess import call
import subprocess
import tqdm

aac_dir = input('aac directory:')
wav_dir = os.sep.join([""] + [x for x in aac_dir.split(os.sep) if x][:-1] + ['wav'])

os.makedirs(wav_dir, exist_ok=True)

for person_dir in tqdm.tqdm(list(glob(os.path.join(aac_dir, '*'))), position=0):
    for video_dir in tqdm.tqdm(list(glob(os.path.join(person_dir, '*'))), position=1, leave=False):
        for utterance in tqdm.tqdm(list(glob(os.path.join(video_dir, '*.m4a'))), position=2, leave=False):
            m4a_name = os.path.basename(utterance)
            wav_path = os.path.join(wav_dir, *os.path.dirname(utterance).split(os.sep)[-2:], m4a_name.replace('.m4a', '.wav'))

            if os.path.exists(wav_path):
                continue

            os.makedirs(os.path.dirname(wav_path), exist_ok=True)

            arguments = [
                'ffmpeg',
                '-loglevel', 'quiet',
                '-i', utterance,
                wav_path,
                ]

            call(arguments)
