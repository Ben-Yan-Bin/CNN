from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.xception import preprocess_input, decode_predictions
import numpy as np
import time

model = Xception(weights='imagenet')

for i, layer in enumerate(model.layers):
    print(i, layer.name)

img_path = 'img/football.jpg'
# img_path = 'img/mercedes.jpg'
img = image.load_img(img_path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

time1 = time.time()
preds = model.predict(x, batch_size=32)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
time2 = time.time()
# print(preds)
pred = decode_predictions(preds, top=5)[0]
for i in pred:
    print('Predicted:', i)
print("%.2f" % (time2-time1), " secs passed.")
