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
from tqdm import tqdm
from emomatch_dataset import EmoMatchDataset
from random import shuffle

data_directory = ''

N = 5000
n_video_samples = 1

ds_p = EmoMatchDataset(data_directory, n_video_samples=n_video_samples)

indices = list(range(len(ds_p)))

shuffle(indices)

indices = indices[:N]

img_features = [None]*N
aud_features = [None]*N
for i, n in enumerate(tqdm(indices)):
    item = ds_p[n]
    features, _ = item
    features = [x.numpy() for x in features]
    features_aud, features_img = features

    img_features[i] = features_img
    aud_features[i] = features_aud

img_features = np.array(img_features)
aud_features = np.array(aud_features)

print(aud_features.shape)

img_features = img_features.transpose([0,1,3,4,2]).reshape(-1, 3)
aud_features = aud_features.transpose([0,1,3,2]).reshape(-1, 512)

img_means = np.mean(img_features, 0)
aud_means = np.mean(aud_features, 0)

img_stds = np.std(img_features, 0)
aud_stds = np.std(aud_features, 0)

print(img_means.shape, aud_means.shape)
print(img_stds.shape, aud_stds.shape)

np.save('img_norms.npy', (img_means, img_stds))
np.save('aud_norms.npy', (aud_means, aud_stds))