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

experiment_name = "exp7_ibn_md"                         # experiment name
experiment_path = "/app/experiments"                    # experiment base path
output_path = "%s/%s" % (experiment_path, experiment_name)

# Model parameters
core_model = "ibn_model"                                # [bn_model, in_model, bn_sn_model, in_sn_model, ibn_model]
d_scales = 3                                            # number of multi discriminator scales

# Data parameters
data_path = "/app/data"                                 # data path for train and test
input_shape = (256, 256)                                # input shape
input_color_mode = 'rgb'                                # input colour space (same as data path content)
output_color_mode = 'lab'                               # output colour space
interpolation = 'nearest'                               # interpolation for reshaping operations
chunk_size = 10000                                      # reading chunk size
samples_rate = 1.                                       # percentage of output samples within data path
shuffle = True                                          # shuffle during training
seed = 42                                               # seed for shuffle operation

# Training parameters
epochs = 200                                            # training epochs
batch_size = 16                                         # batch size for train and validation. batch_size = 1 for test

l1_lambda = 100                                         # l1 weight into global loss function
lr_d = 0.0002                                           # learning rate for discriminator
lr_g = 0.0002                                           # learning rate for generator
beta = 0.5                                              # learning beta

display_step = 10                                       # step size for updating tensorboard log file
plots_per_epoch = 20                                    # prediction logs per epoch
weights_per_epoch = 20                                  # weight checkpoints per epoch

multi_gpu = False                                       # enable multi gpu model
gpus = 2                                                # number of available gpus
workers = 10                                            # number of worker threads
max_queue_size = 10                                     # queue size for worker threads
use_multiprocessing = False                             # use multiprocessing
