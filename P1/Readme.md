# How to use this code
Two CNN models (i.e. [VGG16](https://arxiv.org/abs/1409.1556)[1] and [ResNet-50](https://arxiv.org/abs/1512.03385)[2]) are implemented by 'vgg16.py' and 'resnet.py', respectively.

The following examples demonstrate how to load CNN models (pretrained on ImageNet) using 'vgg16.py' and 'resnet.py', modify their achitectures and fine-tuning model weights for image classification

## VGG16 model
```
from vgg16 import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D,Dropout

baseModel = vgg16(224, 224, 3, 3, include_top=False)
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(4096, activation='relu', name='fc1')(headModel)
headModel = Dense(4096, activation='relu', name='fc2')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(num_classes, activation='softmax')(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False
    
model.summary()
```    
## ResNet-50 model
```
from resnet import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D,Dropout

baseModel = ResNet50(include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(num_classes, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

model.summary()
```
In the above examples, ```num_classes``` is the number of image classes

**NOTE:** To make this code works with 'resnet.py', you need to install **tensorflow-gpu 2.2.0** or **tensorflow 2.2.0**
## Reference
1. Simonyan K., Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition. International Conference on Learning Representations(ICLR), 2015.
2. He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.
