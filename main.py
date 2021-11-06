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

import argparse
import os
from importlib.machinery import SourceFileLoader

from data.data_generator import Imagenet
from models.colorgan import ColorGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ColorGAN main')
    parser.add_argument('-c', '--config', type=str, default=None, help='config file')
    parser.add_argument('-a', '--action', type=str, default=None, help='train or test')
    parser.add_argument('-g', '--gpu', type=str, default=0, help='gpu number')
    args = parser.parse_args()

    assert args.config is not None
    cf = SourceFileLoader('config', args.config).load_module()
    if not cf.multi_gpu: os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    gan = ColorGAN(cf)

    if args.action == "train":
        datagen = Imagenet(cf, mode='train')
        gan.train(datagen)
    elif args.action == "test":
        datagen = Imagenet(cf, mode='val')
        gan.test(datagen)
    else:
        raise ValueError('Invalid action')
