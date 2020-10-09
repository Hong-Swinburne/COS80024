# How to use this code
The CNN model of [ResNet-50](https://arxiv.org/abs/1512.03385)[1]) is implemented by 'resnet.py'.

The following example demonstrates how to load CNN model (pretrained on ImageNet) using 'resnet.py', modify its achitecture for fine-tuning of model weights

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

**NOTE:** To make this code work, you need to install **tensorflow-gpu 2.2.0** or **tensorflow 2.2.0**
## Reference
1. He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.
