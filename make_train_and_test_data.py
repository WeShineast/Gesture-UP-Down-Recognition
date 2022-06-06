import random
import pathlib
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

data_path = pathlib.Path('./dataset')
all_image_paths = list(data_path.glob('*/*'))  
# 所有图片路径的列表
all_image_paths = [str(path) for path in all_image_paths]
# 打散
random.shuffle(all_image_paths)
label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())
# 标签转化为整型值
label_to_index = dict((name, index) for index, name in enumerate(label_names)) 
# 图片和标签对应
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths] 

'''
# 验证一下训练集是否打好标签
i=int(input('Enter a number b/w 0-825'))
plt.imshow(all_image_paths[i],cmap=plt.cm.binary)
maping=['GESTURE DOWN','GESTURE UP']
print(maping[all_image_labels[i]])
plt.show()
'''

img_data_array=[]
for img_path in all_image_paths:
    img=cv.imread(img_path)
    img=cv.resize(img, (128, 128),interpolation = cv.INTER_AREA)
    img=np.array(img)
    img_data_array.append(img)

X=np.array(img_data_array)
y=np.array(all_image_labels)
print(np.shape(X))
# 数据集划分——80%训练集+20%测试集
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y)
print(np.shape(x_train))
print(np.shape(x_test))