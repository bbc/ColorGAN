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

from keras.layers import Concatenate, BatchNormalization
from keras.layers import Conv2D, UpSampling2D, AvgPool2D
from keras.layers import Input, LeakyReLU, Activation
from keras.models import Model
from keras.utils import multi_gpu_model

from .layers.instancenormalization import InstanceNormalization
from .layers.spectralnormalization import ConvSN2D


class _BaseModel:
    def __init__(self, name='base_model',
                 conv_layer=Conv2D):
        self.name = name
        self.conv_layer = conv_layer

    def down_block(self, layer_input, filters, filter_size=4, batch_norm=True, in_norm=False):
        d = self.conv_layer(filters, kernel_size=filter_size, strides=2, padding='same')(layer_input)
        if batch_norm:
            d = BatchNormalization()(d)
        if in_norm:
            d = InstanceNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        return d

    def up_block(self, layer_input, skip_input, filters, filter_size=4, batch_norm=True, in_norm=False):
        u = UpSampling2D(size=2)(layer_input)
        u = self.conv_layer(filters, kernel_size=filter_size, strides=1, padding='same')(u)
        if batch_norm:
            u = BatchNormalization()(u)
        if in_norm:
            u = InstanceNormalization()(u)
        u = Activation('relu')(u)
        u = Concatenate()([u, skip_input])
        return u

    @staticmethod
    def default_disc(input_shape, d_scales, core_model):
        inputs = [Input(shape=input_shape + (3,))]
        for i in range(1, d_scales):
            inputs.append(AvgPool2D(2, padding='same')(inputs[-1]))
        return [core_model(inputs[0], i, s) for s, i in enumerate(inputs)]


class BNModel(_BaseModel):
    def __init__(self, name='bn_model',
                 input_shape=(256, 256),
                 d_scales=1,
                 multi_gpu=True,
                 gpus=2,
                 conv_layer=Conv2D):
        super(BNModel, self).__init__(name=name, conv_layer=conv_layer)
        self.input_shape = input_shape
        self.d_scales = d_scales
        self.multi_gpu = multi_gpu
        self.gpus = gpus

    def generator(self):
        inputs = Input(shape=self.input_shape + (1,))
        d2 = self.down_block(inputs, 64, batch_norm=False)
        d3 = self.down_block(d2, 128)
        d4 = self.down_block(d3, 256)
        d5 = self.down_block(d4, 512)
        d6 = self.down_block(d5, 512)
        d7 = self.down_block(d6, 512)
        d8 = self.down_block(d7, 512)
        u1 = self.up_block(d8, d7, 512)
        u2 = self.up_block(u1, d6, 512)
        u3 = self.up_block(u2, d5, 512)
        u4 = self.up_block(u3, d4, 256)
        u5 = self.up_block(u4, d3, 128)
        u6 = self.up_block(u5, d2, 64)
        u7 = UpSampling2D(size=2)(u6)
        output = self.conv_layer(2, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
        model = Model(inputs, output, name='generator')
        if self.multi_gpu: model = multi_gpu_model(model, gpus=self.gpus)
        return model

    def discriminator(self):
        def _core_model(inputs, x, scale):
            x = self.down_block(x, 64, batch_norm=False)
            x = self.down_block(x, 128)
            x = self.down_block(x, 256)
            x = self.down_block(x, 512)
            x = self.conv_layer(1, kernel_size=4, strides=1, padding='same')(x)
            model = Model(inputs, x, name='discriminator_' + str(scale))
            if self.multi_gpu: model = multi_gpu_model(model, gpus=self.gpus)
            return model

        return self.default_disc(self.input_shape, self.d_scales, _core_model)


class INModel(_BaseModel):
    def __init__(self, name='in_model',
                 input_shape=(256, 256),
                 d_scales=1,
                 multi_gpu=True,
                 gpus=2,
                 conv_layer=Conv2D):
        super(INModel, self).__init__(name=name, conv_layer=conv_layer)
        self.input_shape = input_shape
        self.d_scales = d_scales
        self.multi_gpu = multi_gpu
        self.gpus = gpus

    def generator(self):
        inputs = Input(shape=self.input_shape + (1,))
        d2 = self.down_block(inputs, 64, batch_norm=False)
        d3 = self.down_block(d2, 128, batch_norm=False, in_norm=True)
        d4 = self.down_block(d3, 256, batch_norm=False, in_norm=True)
        d5 = self.down_block(d4, 512, batch_norm=False, in_norm=True)
        d6 = self.down_block(d5, 512, batch_norm=False, in_norm=True)
        d7 = self.down_block(d6, 512, batch_norm=False, in_norm=True)
        d8 = self.down_block(d7, 512, batch_norm=False, in_norm=True)
        u1 = self.up_block(d8, d7, 512, batch_norm=False, in_norm=True)
        u2 = self.up_block(u1, d6, 512, batch_norm=False, in_norm=True)
        u3 = self.up_block(u2, d5, 512, batch_norm=False, in_norm=True)
        u4 = self.up_block(u3, d4, 256, batch_norm=False, in_norm=True)
        u5 = self.up_block(u4, d3, 128, batch_norm=False, in_norm=True)
        u6 = self.up_block(u5, d2, 64, batch_norm=False, in_norm=True)
        u7 = UpSampling2D(size=2)(u6)
        output = self.conv_layer(2, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
        model = Model(inputs, output, name='generator')
        if self.multi_gpu: model = multi_gpu_model(model, gpus=self.gpus)
        return model

    def discriminator(self):
        def _core_model(inputs, x, scale):
            x = self.down_block(x, 64, batch_norm=False)
            x = self.down_block(x, 128, batch_norm=False, in_norm=True)
            x = self.down_block(x, 256, batch_norm=False, in_norm=True)
            x = self.down_block(x, 512, batch_norm=False, in_norm=True)
            x = self.conv_layer(1, kernel_size=4, strides=1, padding='same')(x)
            model = Model(inputs, x, name='discriminator_' + str(scale))
            if self.multi_gpu: model = multi_gpu_model(model, gpus=self.gpus)
            return model

        return self.default_disc(self.input_shape, self.d_scales, _core_model)


class IBNModel(_BaseModel):
    def __init__(self, name='ibn_model',
                 input_shape=(256, 256),
                 d_scales=1,
                 multi_gpu=True,
                 gpus=2,
                 conv_layer=ConvSN2D):
        super(IBNModel, self).__init__(name=name, conv_layer=conv_layer)
        self.input_shape = input_shape
        self.d_scales = d_scales
        self.multi_gpu = multi_gpu
        self.gpus = gpus

    def generator(self):
        inputs = Input(shape=self.input_shape + (1,))
        d2 = self.down_block(inputs, 64, batch_norm=False)
        d3 = self.down_block(d2, 128)
        d4 = self.down_block(d3, 256)
        d5 = self.down_block(d4, 512)
        d6 = self.down_block(d5, 512)
        d7 = self.down_block(d6, 512)
        d8 = self.down_block(d7, 512)
        u1 = self.up_block(d8, d7, 512)
        u2 = self.up_block(u1, d6, 512)
        u3 = self.up_block(u2, d5, 512)
        u4 = self.up_block(u3, d4, 256)
        u5 = self.up_block(u4, d3, 128)
        u6 = self.up_block(u5, d2, 64, batch_norm=False, in_norm=True)
        u7 = UpSampling2D(size=2)(u6)
        output = self.conv_layer(2, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
        model = Model(inputs, output, name='generator')
        if self.multi_gpu: model = multi_gpu_model(model, gpus=self.gpus)
        return model

    def discriminator(self):
        def _core_model(inputs, x, scale):
            x = self.down_block(x, 64, batch_norm=False)
            x = self.down_block(x, 128, batch_norm=False, in_norm=True)
            x = self.down_block(x, 256)
            x = self.down_block(x, 512)
            x = self.conv_layer(1, kernel_size=4, strides=1, padding='same')(x)
            model = Model(inputs, x, name='discriminator_' + str(scale))
            if self.multi_gpu: model = multi_gpu_model(model, gpus=self.gpus)
            return model

        return self.default_disc(self.input_shape, self.d_scales, _core_model)


class BNSNModel(BNModel):
    def __init__(self, name='bn_sn_model',
                 input_shape=(256, 256),
                 d_scales=1,
                 multi_gpu=True,
                 gpus=2):
        super(BNSNModel, self).__init__(name=name,
                                        input_shape=input_shape,
                                        d_scales=d_scales,
                                        multi_gpu=multi_gpu,
                                        conv_layer=ConvSN2D,
                                        gpus=gpus)


class INSNModel(INModel):
    def __init__(self, name='in_sn_model',
                 input_shape=(256, 256),
                 d_scales=1,
                 multi_gpu=True,
                 gpus=2):
        super(INSNModel, self).__init__(name=name,
                                        input_shape=input_shape,
                                        d_scales=d_scales,
                                        multi_gpu=multi_gpu,
                                        conv_layer=ConvSN2D,
                                        gpus=gpus)
