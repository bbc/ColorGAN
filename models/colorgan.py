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

import numpy as np
from keras.layers import Input, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from .models import BNModel, INModel, IBNModel
from .models import BNSNModel, INSNModel
from .test import test_gan
from .train import train_gan


class ColorGAN:
    def __init__(self, cf, name='color_gan'):

        self.name = name
        self.cf = cf

        if self.cf.core_model is 'bn_model':
            model = BNModel(input_shape=self.cf.input_shape, d_scales=self.cf.d_scales,
                            multi_gpu=self.cf.multi_gpu, gpus=self.cf.gpus)
        elif self.cf.core_model is 'in_model':
            model = INModel(input_shape=self.cf.input_shape, d_scales=self.cf.d_scales,
                            multi_gpu=self.cf.multi_gpu, gpus=self.cf.gpus)
        elif self.cf.core_model is 'bn_sn_model':
            model = BNSNModel(input_shape=self.cf.input_shape, d_scales=self.cf.d_scales,
                              multi_gpu=self.cf.multi_gpu, gpus=self.cf.gpus)
        elif self.cf.core_model is 'in_sn_model':
            model = INSNModel(input_shape=self.cf.input_shape, d_scales=self.cf.d_scales,
                              multi_gpu=self.cf.multi_gpu, gpus=self.cf.gpus)
        elif self.cf.core_model is 'ibn_model':
            model = IBNModel(input_shape=self.cf.input_shape, d_scales=self.cf.d_scales,
                             multi_gpu=self.cf.multi_gpu, gpus=self.cf.gpus)
        else:
            raise ValueError('Invalid core model')

        self.generator = model.generator()
        self.discriminator = model.discriminator()
        if not isinstance(self.discriminator, list):
            self.discriminator = [self.discriminator]

        optimizer_d = Adam(self.cf.lr_d, self.cf.beta, decay=0)
        optimizer_g = Adam(self.cf.lr_g, self.cf.beta, decay=0)

        for d in self.discriminator:
            d.trainable = True
            d.compile(optimizer=optimizer_d, loss='mse')
            d.trainable = False

        gan_input = Input(shape=self.cf.input_shape + (1,))
        gan_output = self.generator(gan_input)
        d_input = Concatenate(axis=-1)([gan_input, gan_output])

        self.combined = Model(inputs=gan_input,
                              outputs=[d(d_input) for d in self.discriminator] +
                                      [gan_output])

        if self.cf.multi_gpu: self.combined = multi_gpu_model(self.combined, gpus=self.cf.gpus)
        self.combined.compile(optimizer=optimizer_g,
                              loss_weights=[1] * self.cf.d_scales + [self.cf.l1_lambda],
                              loss=['mse'] * self.cf.d_scales + ['mae'])

        self.dis_real_Y = []
        self.dis_fake_Y = []
        self.gen_real_Y = []

        for d in self.discriminator:
            d.trainable = True
            d.compile(optimizer=optimizer_d, loss='mse')

            rows = d.outputs[0].shape[1].value
            cols = d.outputs[0].shape[2].value
            self.dis_real_Y.append(np.ones([self.cf.batch_size, rows, cols, 1]))
            self.dis_fake_Y.append(np.zeros([self.cf.batch_size, rows, cols, 1]))
            self.gen_real_Y.append(np.ones([self.cf.batch_size, rows, cols, 1]))

    def save_weights(self, weights_path):
        self.generator.save_weights(weights_path)

    def load_weights(self, weights_path):
        self.generator.load_weights(weights_path)

    def train(self, data_generator):
        train_gan(self, data_generator, self.cf)

    def test(self, data_generator):
        test_gan(self, data_generator, self.cf)
