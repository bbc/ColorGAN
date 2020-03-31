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

import numpy as np
from keras.callbacks import TensorBoard
from keras.utils import GeneratorEnqueuer
from keras.utils import generic_utils
from tensorflow import Summary


class _logs_manager:
    def __init__(self, log_path, model, init_step=0):
        if not os.path.exists(log_path): os.makedirs(log_path)
        self.plots_path = os.path.join(log_path, 'plots')
        if not os.path.exists(self.plots_path): os.makedirs(self.plots_path)
        self.tb_callback = TensorBoard(log_path)
        self.tb_callback.set_model(model.combined)
        self.step = 0
        self.val_step = 0,
        self.plot_index = init_step

    def update(self, progbar, names, values, val_names=None, val_values=None, display_step=10):
        logs_list = []
        for name, value in zip(names, values):
            logs_list.append((name, value))
            if (self.step + 1) % display_step == 0 and self.step != 0:
                summary = Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = name
                self.tb_callback.writer.add_summary(summary, ((self.step + 1) // display_step) - 1)
                self.tb_callback.writer.flush()

        if val_names is not None and val_values is not None:
            for name, value in zip(val_names, val_values):
                logs_list.append((name, value))
                summary = Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value
                summary_value.tag = name
                self.tb_callback.writer.add_summary(summary, self.val_step)
                self.tb_callback.writer.flush()
            self.val_step += 1

        progbar.add(1, values=logs_list)
        self.step += 1

    def save_plots(self, epoch, step, batch_fake, batch_real):
        np.save(os.path.join(self.plots_path, 'p%d_e%d_s%d_fake' %
                             (self.plot_index, epoch, step)), batch_fake[:16])
        np.save(os.path.join(self.plots_path, 'p%d_e%d_s%d_real' %
                             (self.plot_index, epoch, step)), batch_real[:16])
        self.plot_index += 1


class _checkpoint:
    def __init__(self, path):
        self.weights_path = os.path.join(path, 'weights')
        if not os.path.exists(self.weights_path): os.makedirs(self.weights_path)
        self.monitor = np.inf
        self.weight_index = 0

    def save_weights(self, model, epoch, step):
        name = os.path.join(self.weights_path, 'weights_w%d_e%d_s%d.hdf5' %
                            (self.weight_index, epoch, step))
        model.save_weights(name)
        self.weight_index += 1


def train_gan(gan, data_generator, cf):
    check = _checkpoint(cf.output_path)
    logs = _logs_manager(cf.output_path, gan)

    train_datagen = GeneratorEnqueuer(data_generator, use_multiprocessing=cf.use_multiprocessing)
    train_datagen.start(cf.workers, cf.max_queue_size)
    train_generator = train_datagen.get()

    for epoch in range(cf.epochs):
        print('Epoch %d/%d' % (epoch + 1, cf.epochs))
        progbar = generic_utils.Progbar(data_generator.nb_steps)

        # Training loop
        for step in range(data_generator.nb_steps):
            batch_real = next(train_generator)
            batch_gray = np.expand_dims(batch_real[:, :, :, 0], -1)
            batch_chroma = batch_real[:, :, :, 1:]
            batch_fake = gan.combined.predict(batch_gray)[-1]

            dis_real_X = np.concatenate((batch_gray, batch_chroma), axis=-1)
            dis_fake_X = np.concatenate((batch_gray, batch_fake), axis=-1)

            d_loss = []
            for d, real_Y, fake_Y in zip(gan.discriminator, gan.dis_real_Y, gan.dis_fake_Y):
                d_real = d.train_on_batch(dis_real_X, real_Y)
                d_fake = d.train_on_batch(dis_fake_X, fake_Y)
                d_loss.append(0.5 * np.add(d_real, d_fake))

            g_loss = gan.combined.train_on_batch(batch_gray, gan.gen_real_Y + [batch_chroma])

            if step % (data_generator.nb_steps // cf.plots_per_epoch) == 0 and step != 0:
                logs.save_plots(epoch, step, data_generator.decoder(dis_fake_X), data_generator.decoder(batch_real))

            if step % (data_generator.nb_steps // cf.weights_per_epoch) == 0 and step != 0 and epoch != 0:
                check.save_weights(gan, epoch, step)

            d_names = ['d_loss_%d' % i for i in range(cf.d_scales)]
            g_names = ['g_cgan_%d' % i for i in range(cf.d_scales)]

            logs.update(
                names=['d_loss'] + d_names + ['g_loss'] + g_names + ['g_l1'],
                values=[sum(d_loss)] + d_loss + [g_loss[0]] + g_loss[1:-1] + [g_loss[-1]],
                progbar=progbar, display_step=cf.display_step)

    train_datagen.stop()
