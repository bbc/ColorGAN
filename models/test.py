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

import os

import cv2
import keras.backend as K
import numpy as np
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers import Input, Lambda, Average
from keras.metrics import mean_absolute_error
from keras.models import Model
from keras.utils import GeneratorEnqueuer
from skimage.color import rgb2lab


def resize(batch, shape=(224, 224)):
    out = []
    for sample in batch:
        out.append(cv2.resize(sample, shape, interpolation=cv2.INTER_CUBIC))
    return np.array(out)


def get_layer(model, name):
    for l in model.layers:
        if l.name == name: return l


def l1_loss(real, fake):
    def loss(x):
        A, B = x
        return K.mean(mean_absolute_error(A, B))

    return Lambda(loss, output_shape=(1,))([real, fake])


def perceptual_model():
    vgg = VGG19(include_top=True, weights='imagenet', classes=1000)

    layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    layers = [get_layer(vgg, i).output for i in layer_names]
    model = Model(vgg.inputs, layers)

    inputs_fake = Input((224, 224, 3))
    inputs_real = Input((224, 224, 3))

    maps_fake = model(inputs_fake)
    maps_real = model(inputs_real)

    means = [l1_loss(r, f) for r, f in zip(maps_real, maps_fake)]
    l1 = Average()(means)

    return Model([inputs_real, inputs_fake], l1)


def compute_prior_dist(data_generator, cf):
    prior_path = os.path.join(cf.output_path, 'priors')
    if not os.path.exists(prior_path): os.makedirs(prior_path)

    test_datagen = GeneratorEnqueuer(data_generator, use_multiprocessing=cf.use_multiprocessing)
    test_datagen.start(cf.workers, cf.max_queue_size)
    test_generator = test_datagen.get()

    a_cum_dist = np.zeros(219)
    b_cum_dist = np.zeros(219)

    for _ in tqdm.tqdm(range(data_generator.samples)):
        batch_real, _ = next(test_generator)
        rgb_batch = data_generator.decoder(batch_real)

        for image in rgb_batch:
            image = rgb2lab(image)
            a_cum_dist += np.histogram(image[..., 1], bins=np.arange(-110., 110.))[0]
            b_cum_dist += np.histogram(image[..., 2], bins=np.arange(-110., 110.))[0]

    a_dist = np.log(a_cum_dist / data_generator.samples)
    b_dist = np.log(b_cum_dist / data_generator.samples)

    a_dist[a_dist == -np.inf] = np.inf
    a_dist[a_dist == np.inf] = a_dist.min()
    b_dist[b_dist == -np.inf] = np.inf
    b_dist[b_dist == np.inf] = b_dist.min()

    np.save(os.path.join(prior_path, 'a_dist.npy'), a_dist)
    np.save(os.path.join(prior_path, 'b_dist.npy'), b_dist)


def test_gan(gan, data_generator, cf):
    results_path = os.path.join(cf.output_path, 'results')
    if not os.path.exists(results_path): os.makedirs(results_path)

    l1_cum = 0
    psnr_cum = 0
    perc_cum = 0
    a_cum_dist = np.zeros(219)
    b_cum_dist = np.zeros(219)

    perception = perceptual_model()
    gan.load_weights(cf.weights_path)

    test_datagen = GeneratorEnqueuer(data_generator, use_multiprocessing=cf.use_multiprocessing)
    test_datagen.start(cf.workers, cf.max_queue_size)
    test_generator = test_datagen.get()

    for _ in tqdm.tqdm(range(data_generator.samples)):
        batch_real, _ = next(test_generator)
        batch_gray = np.expand_dims(batch_real[..., 0], -1)
        batch_pred = gan.predict(batch_gray)
        batch_fake = np.concatenate((batch_gray, batch_pred), axis=-1)

        # compute color distribution
        real_rgb = data_generator.decoder(batch_real)
        fake_rgb = data_generator.decoder(batch_fake)
        real_lab, fake_lab = [], []
        for fake, real in zip(fake_rgb, real_rgb):
            fake = rgb2lab(fake)
            fake_lab.append(fake)
            real_lab.append(rgb2lab(real))
            a_cum_dist += np.histogram(fake[..., 1], bins=np.arange(-110., 110.))[0]
            b_cum_dist += np.histogram(fake[..., 2], bins=np.arange(-110., 110.))[0]

        fake_lab = np.array(fake_lab)
        real_lab = np.array(real_lab)

        # compute l1
        l1_cum += np.abs(np.subtract(fake_lab[..., 1:], real_lab[..., 1:])).mean()

        # compute psnr
        mse = np.square(np.subtract(fake_lab[..., 1:], real_lab[..., 1:])).mean()
        psnr_cum += np.sum(20 * np.log10(255. / np.sqrt(mse)))

        # compute perceptual loss
        real_rgb = resize(real_rgb)
        fake_rgb = resize(fake_rgb)
        perc_cum += sum(perception.predict([preprocess_input(i) for i in [real_rgb, fake_rgb]]))

    l1 = l1_cum / data_generator.samples
    psnr = psnr_cum / data_generator.samples
    perc_loss = perc_cum / data_generator.samples

    with open(results_path + '/logs.txt', 'a') as f:
        f.write('experiment, l1, psnr, perc loss\n')
        f.write(cf.experiment_name + ', %f, %f, %f\n' % (l1, psnr, perc_loss))

    a_dist = np.log(a_cum_dist / data_generator.samples)
    b_dist = np.log(b_cum_dist / data_generator.samples)

    a_dist[a_dist == -np.inf] = np.inf
    a_dist[a_dist == np.inf] = a_dist.min()
    b_dist[b_dist == -np.inf] = np.inf
    b_dist[b_dist == np.inf] = b_dist.min()

    np.save(os.path.join(results_path, 'a_dist.npy'), a_dist)
    np.save(os.path.join(results_path, 'b_dist.npy'), b_dist)
