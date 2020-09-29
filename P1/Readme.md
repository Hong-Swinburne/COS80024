# How to use this code
Two CNN models (i.e. VGG16 and ResNet-50) are implemented in vgg16.py and 'resnet.py', respectively.

The following examples demonstrate how to load CNN models (pretrained on ImageNet) using 'vgg16.py' and 'resnet.py', modify their achitectures and fine-tuning model weights for image classification

## VGG16 model
```
from vgg16 import vgg16
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
baseModel = ResNet50(weights="imagenet", include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))
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
