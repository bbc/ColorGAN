# Copyright 2020 BBC Research & Development
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import csv
import os
import threading

import keras.backend as K
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from skimage.color import rgb2lab, lab2rgb

SYNSET_WORDS = 'synset_words.txt'  # Synset for ILSVRC 2012 dataset. Modify according to the database version.


class Imagenet:
    def __init__(self, cf,
                 mode='train',
                 get_labels=False,
                 get_categorical=False):

        assert mode in ['train', 'test', 'val'], 'Invalid mode'

        self.mode = mode
        self.path = os.path.join(cf.data_path, mode)
        self.batch_size = 1 if mode is "test" else cf.batch_size
        self.data_shape = cf.input_shape

        self._input_color_mode = cf.input_color_mode
        self._output_color_mode = cf.output_color_mode
        self._interpolation = cf.interpolation
        self._chunk_size = cf.chunk_size // cf.batch_size * cf.batch_size
        self._seed = cf.seed
        self.shuffle = cf.shuffle
        self._lock = threading.Lock()
        self._get_labels = get_labels
        self._get_categorical = get_categorical

        self._data_format = K.image_data_format()
        self._dtype = K.floatx()

        if self._input_color_mode is 'rgb':
            if self._data_format == 'channels_last':
                self.image_shape = self.data_shape + (3,)
            else:
                self.image_shape = (3,) + self.data_shape
        else:
            if self._data_format == 'channels_last':
                self.image_shape = self.data_shape + (1,)
            else:
                self.image_shape = (1,) + self.data_shape

        self.batch_index = 0
        self._chunk_index = 0
        self._total_batches_seen = 0

        self.nb_classes = 1000
        self._filenames = self._process_data(cf.samples_rate)
        self._index_generator = self._flow_index()
        self.val_batch = self._get_val_batch()
        self._step = 0

    def _process_data(self, samples_rate, ext=('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'tif', 'tiff')):
        ext = tuple('.%s' % e for e in ((ext,) if isinstance(ext, str) else ext))
        names = [os.path.join(root, f) for root, _, files in os.walk(self.path)
                 for f in files if f.lower().endswith(ext)]
        
        self._chunk_size = np.minimum(int(len(names) * samples_rate), self._chunk_size)
        
        self.samples = int(len(names) * samples_rate) // self._chunk_size * self._chunk_size
        self.nb_steps = self.samples // self.batch_size
        self._nb_chunks = self.samples // self._chunk_size
        print('Found %d images belonging to %d classes.' % (self.samples, self.nb_classes))
        names = np.array(names)
        np.random.shuffle(names)
        return np.array_split(names[:self.samples], self._nb_chunks)

    def _reset(self):
        self.batch_index = 0

    def epoch_end(self):
        self._reset()
        self._chunk_index = 0
        self._step = 0

    def _flow_index(self):
        while True:
            self._reset()
            for self._chunk_index in range(self._nb_chunks):
                np.random.seed(self._seed + self._total_batches_seen)
                index_array = np.random.permutation(self._chunk_size) \
                    if self.shuffle else np.arange(self._chunk_size)
                for self._step in range(self._chunk_size // self.batch_size):
                    self.batch_index += 1 if self.batch_index + 1 < self.samples else 0
                    self._total_batches_seen += 1
                    yield index_array[self._step * self.batch_size:(self._step + 1) * self.batch_size]

    def _get_batch(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self._dtype)
        names = self._filenames[self._chunk_index]
        for i, j in enumerate(index_array):
            img = load_img(names[j],
                           color_mode=self._input_color_mode,
                           target_size=self.image_shape,
                           interpolation=self._interpolation)
            sample = img_to_array(img, data_format=self._data_format)

            if hasattr(img, 'close'):
                img.close()

            sample = rgb2lab(sample / 255.0)
            sample = np.stack([sample[..., 0] / 50 - 1,
                               sample[..., 1] / 110,
                               sample[..., 2] / 110], axis=-1)
            batch_x[i] = sample

        if self._get_labels:
            batch_y = np.array([int(n.split(self.mode + '/')[1].split('/')[0]) for n in names[index_array]])
            if self._get_categorical:
                batch_y = to_categorical(batch_y, num_classes=self.nb_classes)
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        with self._lock:
            index_array = next(self._index_generator)
        return self._get_batch(index_array)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_val_batch(self):
        return self.decoder(self._get_batch(np.random.randint(self._chunk_size, size=self.batch_size)))

    @staticmethod
    def encoder(batch_rgb):
        batch_lab = []
        for img_rgb in batch_rgb:
            img_lab = rgb2lab(img_rgb / 255.0)
            img_lab = np.stack([img_lab[..., 0] / 50 - 1,
                                img_lab[..., 1] / 110,
                                img_lab[..., 2] / 110], axis=-1)
            batch_lab.append(img_lab)
        return np.array(batch_lab)

    @staticmethod
    def decoder(batch_lab):
        batch_rgb = []
        for img_lab in batch_lab:
            img_lab = np.stack([(img_lab[..., 0] + 1) / 2 * 100,
                                img_lab[..., 1] * 110,
                                img_lab[..., 2] * 110], axis=-1)
            img_rgb = lab2rgb(img_lab) * 255
            batch_rgb.append(np.expand_dims(img_rgb, 0))
        return np.uint8(np.concatenate(batch_rgb, 0))

    def decode_label(self, label_idx):
        label_key = [*self._labels][label_idx]
        return label_key, self._labels[label_key]
