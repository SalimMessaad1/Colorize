# By salm messaad 
# https://www.linkedin.com/in/salimmessaad
import cv2
import numpy as np

# model
prototxt = 'colorization_deploy_v2.prototxt'
model = 'colorization_release_v2.caffemodel'
pts_in_hull = 'pts_in_hull.npy'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

pts = np.load(pts_in_hull)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Loading image
original_image = cv2.imread('123.jpg')

# Check test
if original_image is None:
    print("Error: Could not load image, Double check again for path or name of image sir")
    exit()

scaled_image = original_image.astype("float32") / 255.0
lab_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2Lab)

resized = cv2.resize(lab_image, (224, 224))
L = resized[:, :, 0] - 50  

blob = cv2.dnn.blobFromImage(L)
net.setInput(blob)
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (original_image.shape[1], original_image.shape[0]))
L = lab_image[:, :, 0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized_image = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
colorized_image = (colorized_image * 255).astype("uint8")

height = max(original_image.shape[0], colorized_image.shape[0])
original_resized = cv2.resize(original_image, (int(original_image.shape[1] * height / original_image.shape[0]), height))
colorized_resized = cv2.resize(colorized_image, (int(colorized_image.shape[1] * height / colorized_image.shape[0]), height))

combined_image = np.hstack((original_resized, colorized_resized))
max_width = 1200 
if combined_image.shape[1] > max_width:
    scale_factor = max_width / combined_image.shape[1]
    new_width = max_width
    new_height = int(combined_image.shape[0] * scale_factor)
    combined_image = cv2.resize(combined_image, (new_width, new_height))
#and in last display
cv2.imshow('Original vs Colorized', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
