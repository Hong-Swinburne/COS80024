# How to use this code
Two CNN models (i.e. [VGG16](https://arxiv.org/abs/1409.1556)[1] and [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)[2]) are implemented by 'vgg16.py' and 'alexnet.py', respectively.

The following examples demonstrate how to load CNN models using 'vgg16.py' and 'alexnet.py' and modify their achitectures (if neccessary) for training.
## VGG16 model
```
from vgg16 import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization

baseVGG16 = vgg16(IMG_SIZE, IMG_SIZE, 3, num_classes, include_top=False, weights=None)
baseModel = Model(inputs=baseVGG16.input, outputs=baseVGG16.get_layer('block5_pool').output)
headModel = GlobalAveragePooling2D()(baseModel.output)
headModel = Dense(4096, activation='relu', name='fc1')(headModel)
headModel = BatchNormalization()(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(4096, activation='relu', name='fc2')(headModel)
headModel = BatchNormalization()(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(numLabels, activation='softmax')(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
    
model.summary()
```    
## AlexNet model
```
from alexnet import AlexNet
from tensorflow.keras.models import Model

model = AlexNet(image_height=IMG_SIZE, image_width=IMG_SIZE, channels=3, NUM_CLASSES=num_classes)

model.summary()
```
In the above examples, ```num_classes``` is the number of image classes, and ```IMG_SIZE``` is the size of images that input to the CNN models. In this project, ```IMG_SIZE=48``` is recommended for traffic sign images.  

**NOTE:** To make this code work, you need to install **tensorflow-gpu 2.0 +** or **tensorflow 2.0**+
## Reference
1. Simonyan K., Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition. International Conference on Learning Representations(ICLR), 2015.
2. Krizhevsky A., Sutskever I., Hinton G. E. ImageNet Classification with Deep Convolutional Neural Networks. Advances in Neural Information Processing Systems (NIPS), 2012.
