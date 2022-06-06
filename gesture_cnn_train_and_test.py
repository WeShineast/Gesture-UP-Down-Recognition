import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from make_train_and_test_data import x_train,x_test,y_train,y_test
# x_train:Training images y_train:The labels of Training images
# x_test:Testing images y_test:The labels of Testing images

model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),input_shape=(128,128,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

# loss:损失函数-交叉熵 optimizer:优化器-adam metrics:评价函数
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# 图像增强预处理
train_datagen=ImageDataGenerator(
    rescale=1.0/255, #图像放缩
    rotation_range=40, #角度随机旋转
    width_shift_range=0.2, #水平方向位置随机左右平移
    height_shift_range=0.2, #竖直方向位置随机上下平移
    shear_range=0.2, #随机错切变换
    horizontal_flip=True, #随机对图片执行水平方向翻转操作
    fill_mode='nearest' #填充模式，默认为最近原则
)

#训练
history=model.fit(
    train_datagen.flow(
        x_train,
        y_train,
        batch_size=16, 
        shuffle=True
    ),
    epochs=15,#训练模型的迭代数
    verbose=1,
    )

# 测试
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print("测试集的准确度", test_acc)

# 保存模型参数
model.save('gesture_model.h5')
