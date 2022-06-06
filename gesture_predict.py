import numpy as np
import cv2 as cv
import tensorflow as tf

model=tf.keras.models.load_model('gesture_model.h5')
maping=['GESTURE DOWN','GESTURE UP']

vid=cv.VideoCapture(0)
while(True):
    ret,frame=vid.read()
    cv.imshow("screen",frame)
    if cv.waitKey(1) == ord('q'):
        frame = cv.resize(frame, (1920, 1080))
        cv.imwrite('pred.jpg', frame)
        break
vid.release()
cv.destroyWindow('screen')
    

img=cv.imread('./pred.jpg')
img=cv.resize(img, (128, 128),interpolation = cv.INTER_AREA)
img=img.reshape(1,128,128,3)
img=np.array(img)

pred=model.predict(img)
if pred>0.5:
    print("The image is predicted to be",maping[1])
else:
    print("The image is predicted to be",maping[0])
