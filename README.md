# End-to-End Conditional GAN-based Architectures for Image Colourisation

| ![Marc Górriz][MarcGorriz-photo]  |  ![Marta Mrak][MartaMrak-photo] | ![Alan F. Smeaton][AlanFmeaton-photo]  | ![Noel E. O’Connor][NoelEOConnor-photo]  |
|:-:|:-:|:-:|:-:|
| [Marc Górriz][MarcGorriz-web]  | [Marta Mrak][MartaMrak-web] | [Alan F. Smeaton][AlanFmeaton-web] | [Noel E. O’Connor][NoelEOConnor-web] |

[MarcGorriz-web]: https://www.bbc.co.uk/rd/people/marc-gorriz-blanch
[MartaMrak-web]: https://www.bbc.co.uk/rd/people/marta-mrak
[AlanFmeaton-web]: https://www.insight-centre.org/users/alan-smeaton
[NoelEOConnor-web]: https://github.com/marc-gorriz/ColorGAN/blob/master/authors/NoelEOConnor.jpg

[MarcGorriz-photo]: https://raw.githubusercontent.com/bbc/ColorGAN/blob/master/authors/MarcGorriz.jpg
[MartaMrak-photo]: https://raw.githubusercontent.com/bbc/ColorGAN/master/authors/MartaMrak.jpg
[AlanFmeaton-photo]: https://raw.githubusercontent.com/bbc/ColorGAN/master/authors/AlanFSmeaton.jpg
[NoelEOConnor-photo]: https://raw.githubusercontent.com/bbc/ColorGAN/blob/master/authors/NoelEOConnor.jpg

A joint collaboration between:

| ![logo-bbc] | ![logo-dcu] | ![logo-insight] |
|:-:|:-:|:-:|
| [BBC Research & Development][bbc-web] | [Dublin City University (DCU)][dcu-web] | [Insight Centre for Data Analytics][insight-web] |

[bbc-web]: https://www.bbc.co.uk/rd
[insight-web]: https://www.insight-centre.org/ 
[dcu-web]: http://www.dcu.ie/

[logo-bbc]: https://github.com/marc-gorriz/ColorGAN/blob/master/logos/bbc.png  "BBC Research & Development"
[logo-insight]: https://github.com/marc-gorriz/ColorGAN/blob/master/logos/insight.jpg "Insight Centre for Data Analytics"
[logo-dcu]: https://github.com/marc-gorriz/ColorGAN/blob/master/logos/dcu.png "Dublin City University"

## Abstract
In this work recent advances in conditional adversarial networks are investigated to develop an end-to-end architecture based on Convolutional Neural Networks (CNNs) to directly map realistic colours to an input greyscale image. Observing that existing colourisation methods sometimes exhibit a lack of colourfulness, this work proposes a method to improve colourisation results. In particular, the method uses Generative Adversarial Neural Networks (GANs) and focuses on improvement of training stability to enable better generalisation in large multi-class image datasets. Additionally, the integration of instance and batch normalisation layers in both generator and discriminator is introduced to the popular U-Net architecture, boosting the network capabilities to generalise the style changes of the content. The method has been tested using the [ILSVRC 2012 dataset](http://image-net.org/challenges/LSVRC/2012/), achieving improved automatic colourisation results compared to other methods based on GANs.

![visualisation-fig]

[visualisation-fig]: https://github.com/marc-gorriz/ColorGAN/blob/master/logos/visualisation.png

## Publication
2019 IEEE 21st International Workshop on Multimedia Signal Processing (MMSP). Find the paper discribing our work on [IEEE Xplore](https://ieeexplore.ieee.org/document/8901712) and [arXiv](https://arxiv.org/abs/1908.09873).

Please cite with the following Bibtex code:
```
@inproceedings{blanch2019end,
  title={End-to-End Conditional GAN-based Architectures for Image Colourisation},
  author={Blanch, Marc G{\'o}rriz and Mrak, Marta and Smeaton, Alan F and O'Connor, Noel E},
  booktitle={2019 IEEE 21st International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
```
## How to use

### Dependencies

The model is implemented in [Keras](https://github.com/fchollet/keras/tree/master/keras), which at its time is developed over [TensorFlow](https://www.tensorflow.org). Also, this code should be compatible with Python 3.6.

```
pip install -r https://github.com/marc-gorriz/ColorGAN/blob/master/requeriments.txt
```
Import the Open Source libraries for instance and spectral normalistion. Refer to ```models/layers``` directory

### Prepare data
Training examples are generated from the ImageNet dataset, particularly from the 1,000 synsets selected for the [ImageNet Large Scale Visual Recognition Challenge 2012](http://www.image-net.org/challenges/LSVRC/2012/). Samples are selected from the reduced validation set, containing 50,000 RGB images uniformly distributed as 50 images per class. The test dataset is created by randomly selecting 10 images per class from the training set, generating up to 10,000 examples. All images are resized to 256×256 pixels and converted to the CIE Lab colour space.

Make sure the data path has the following tree structure:
```
-data
 |
 ---- train
 |    |
 |    ---- 0
 |    |    |---- img0.png
 |    |    | …
 |    |    |---- img49.png
 |    | …
 |    ---- 999
 |    |    |---- img0.png
 |    |    | …
 |    |    |---- img49.png
 |
 ---- test
 |    |
 |    ---- 0
 |    |    |---- img0.png
 |    |    | …
 |    |    |---- img9.png
 |    | …
 |    ---- 999
 |    |    |---- img0.png
 |    |    | …
 |    |    |---- img9.png
```

### Launch an experiment
* Make a new configuration file based on the available templates and save it into the ```config``` directory.
Make sure to launch all the processes over GPU.

* To train a new model, run  ```python main.py --config config/[config file].py --action train```.

## Acknowledgements
This work has been conducted within the project
JOLT. This project is funded by the European Union’s Horizon 2020 research
and innovation programme under the Marie Skłodowska Curie grant agreement No 765140.

| ![JOLT-photo] | ![EU-photo] |
|:-:|:-:|
| [JOLT Project](JOLT-web) | [European Comission](EU-web) |


[JOLT-photo]: https://github.com/marc-gorriz/ColorGAN/blob/master/logos/jolt.png "JOLT"
[EU-photo]: https://github.com/marc-gorriz/ColorGAN/blob/master/logos/eu.png "European Comission"


[JOLT-web]: http://joltetn.eu/
[EU-web]: https://ec.europa.eu/programmes/horizon2020/en

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/marc-gorriz/ColorGAN/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:marc.gorrizblanch@bbc.co.uk>.
