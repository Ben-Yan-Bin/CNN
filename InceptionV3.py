from keras.applications.inception_v3 import InceptionV3 as iv3
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import time

model = iv3(weights='imagenet')

img_path = 'img/burger.jpg'
# img_path = 'img/mercedes.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

time1 = time.time()
preds = model.predict(x, batch_size=32)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
time2 = time.time()
pred = decode_predictions(preds, top=5)[0]
for i in pred:
    print('Predicted:', i)
print("%.2f" % (time2-time1), " secs passed.")
