# How to use this code
The CNN model of [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)[1]) is implemented by 'alexnet.py'.

The following example demonstrates how to load CNN model using 'alexnet.py' and modify its achitectures (if neccessary) for training.

```
from alexnet import AlexNet
from tensorflow.keras.models import Model

model = AlexNet(image_height=IMG_SIZE, image_width=IMG_SIZE, channels=3, NUM_CLASSES=num_classes)

model.summary()
```
In the above examples, ```num_classes``` is the number of image classes, and ```IMG_SIZE``` is the size of images that input to the CNN model. In this project, ```IMG_SIZE=48``` is recommended for traffic sign images.  

**NOTE:** To make this code work, you need to install **tensorflow-gpu 2.0 +** or **tensorflow 2.0**+
## Reference
1. Krizhevsky A., Sutskever I., Hinton G. E. ImageNet Classification with Deep Convolutional Neural Networks. Advances in Neural Information Processing Systems (NIPS), 2012.
