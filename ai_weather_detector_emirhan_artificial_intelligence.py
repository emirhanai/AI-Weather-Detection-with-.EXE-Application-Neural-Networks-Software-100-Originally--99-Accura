import cv2
import numpy as np
from tensorflow.keras.models import load_model

cam = cv2.VideoCapture(1)

cv2.namedWindow("Weather Detection AI Software - Emirhan BULUT")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "prediction_image_3.jpg"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        break

cam.release()
cv2.destroyAllWindows()

img_name = cv2.imread("prediction_image_3.jpg")

img_name = img_name / 255
img_name = cv2.resize(img_name, (128, 128))
img_name = np.reshape(img_name, [1, 128, 128, 3])

model = load_model("model_emirhan.h5")

prediction = model.predict(img_name)

font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 30)
fontScale = 1
color = (228, 31, 36)
thickness = 4

print(prediction)
if "{0}".format(prediction[0].max()) == "{0}".format(prediction[0][0]):
    prediction = "Prediction: Cloudy"
elif "{0}".format(prediction[0].max()) == "{0}".format(prediction[0][1]):
    prediction = "Prediction: Rainy"
elif "{0}".format(prediction[0].max()) == "{0}".format(prediction[0][2]):
    prediction = "Prediction: Shine"
elif "{0}".format(prediction[0].max()) == "{0}".format(prediction[0][3]):
    prediction = "Prediction: Snowy"
elif "{0}".format(prediction[0].max()) == "{0}".format(prediction[0][4]):
    prediction = "Prediction: Sunrise"

img_name = cv2.imread("prediction_image_3.jpg")

cv2.putText(img_name, prediction, org, font,
            fontScale, color, thickness, cv2.LINE_4)

text = "Weather AI Detection"
text1 = "by Emirhan BULUT"

cv2.putText(img_name, text, (10, 60), font,
            fontScale, color, thickness, cv2.LINE_4)

cv2.putText(img_name, text1, (10, 90), font,
            fontScale, color, thickness, cv2.LINE_4)

cv2.imwrite("prediction_image_3.jpg", img_name)