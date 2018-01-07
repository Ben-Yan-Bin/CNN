
from keras.applications.inception_resnet_v2 import InceptionResNetV2 as IRV2

model = IRV2(include_top=False)
for i, layer in enumerate(model.layers):
    print(i, layer.name)
