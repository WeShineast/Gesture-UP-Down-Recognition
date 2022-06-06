import cv2
import numpy as np
import time
from time import sleep
from pynq.overlays.base import BaseOverlay


def dataload(path):
    img=cv2.imread(path,cv2.IMREAD_UNCHANGED)
    img=cv2.resize(img,(128,128),interpolation=cv2.INTER_LINEAR)
    img=np.array(img).reshape(128,128,3)/255.0
    return img.astype(np.float32)
def decode_prediction(preds,labels):
    index = np.argsort(-preds).reshape(-1)[0]
    label = labels[index]
    prob = preds.reshape(-1)[index]
    return label,prob

img_path = 'test_down.jpg'
#labels_path = 'labels.npy'
pb_file = 'model.pb'
net = cv2.dnn.readNet(pb_file)
if net.empty():
    print('No layers in net')
frame = dataload(img_path)
time_start=time.time()
net.setInput(cv2.dnn.blobFromImage(frame,swapRB=False, crop=False))
pred=net.forward()
pred=(pred>0.5)
print(pred)

base = BaseOverlay("base.bit")

Delay = 0.3
color_up = 2
color_down = 1
rgbled_position_up = 4 #绿灯
rgbled_position_down = 5 #蓝灯
pred=0 #pred=1向上 pred=0向下

while (base.buttons[0].read()==0):
    if (pred==1):
        base.rgbleds[rgbled_position_up].write(color_up)
        sleep(Delay)
        
    elif (pred==0):
        base.rgbleds[rgbled_position_down].write(color_down)
        sleep(Delay)       
    
print('End of this demo ...')
base.rgbleds[rgbled_position_up].off()
base.rgbleds[rgbled_position_down].off()