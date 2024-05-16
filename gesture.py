"""
Created on Mon May 13 01:01:43 2024

author: leidaqian
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils

from keras import backend as K

if K.backend() == 'tensorflow':
    import tensorflow
    # K.set_image_dim_ordering('tf')
else:
    import theano
    # K.set_image_dim_ordering('th')

# K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')

import numpy as np
# import matplotlib.pyplot as plt
import os

from PIL import Image
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json

import cv2
import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

#input img参数
img_rows, img_cols = 200, 200

# channels的数量
# 对于灰度，使用 1 个值，对于彩色图像，使用 3（R、G、B 通道）
img_channels = 1

# Batch_size 训练
batch_size = 32

## 4 种手势（确定、和平、出拳、停止）
nb_classes = 5

# 要训练的 epoch 数
nb_epoch = 15  # 25

# 使用卷积滤波器的总数
nb_filters = 32
# 最大池化
nb_pool = 2
# 卷积核的大小
nb_conv = 3

#  data
path = "./"
path1 = "./gestures"

## Path2是用于训练模型的文件夹。
path2 = './Lei数据集'

WeightFileName = []

# outputs
output = ["OK", "NOTHING", "PEACE", "PUNCH", "STOP"]
# output = ["PEACE", "STOP", "THUMBSDOWN", "THUMBSUP"]

jsonarray = {}


# %%
def update(plot):
    global jsonarray
    h = 450
    y = 30
    w = 45
    font = cv2.FONT_HERSHEY_SIMPLEX


    for items in jsonarray:
        mul = (jsonarray[items]) / 100
        # mul = random.randint(1,100) / 100
        cv2.line(plot, (0, y), (int(h * mul), y), (255, 0, 0), w)
        cv2.putText(plot, items, (0, y + 5), font, 0.7, (0, 255, 0), 2, 1)
        y = y + w + 30

    return plot


# %% 用于 debug 追踪
def debugme():
    import pdb
    pdb.set_trace()


# 将彩色 img 转换为灰度 img
# 将图像从 path1 复制到 path2
def convertToGrayImg(path1, path2):
    listing = os.listdir(path1)
    for file in listing:
        if file.startswith('.'):
            continue
        img = Image.open(path1 + '/' + file)
        # img = img.resize((img_rows,img_cols))
        grayimg = img.convert('L')
        grayimg.save(path2 + '/' + file, "PNG")


def modlistdir(path, pattern=None):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #  这个检查是忽略任何隐藏的文件/文件夹
        if pattern == None:
            if name.startswith('.'):
                continue
            else:
                retlist.append(name)
        elif name.endswith(pattern):
            retlist.append(name)

    return retlist


# Load CNN model
def loadCNN(bTraining=False):
    global get_output
    model = Sequential()

    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                     padding='valid',
                     input_shape=(img_channels, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # Model summary
    model.summary()
    # Model conig details
    model.get_config()

    if not bTraining:
        # 列出当前目录中可用的所有权重文件
        WeightFileName = modlistdir('.', '.hdf5')
        if len(WeightFileName) == 0:
            print(
                'Error: No pretrained weight file found. Please either train the model or download one from the https://github.com/asingh33/CNNGestureRecognizer')
            return 0
        else:
            print('Found these weight files - {}'.format(WeightFileName))
        # Load pretrained weights
        w = int(input("Which weight file to load (enter the INDEX of it, which starts from 0): "))
        fname = WeightFileName[int(w)]
        print("loading ", fname)
        model.load_weights(fname)

    # refer the last layer here
    layer = model.layers[-1]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])

    return model


# 此函数根据输入图像进行猜测工作
def guessGesture(model, img):
    global output, get_output, jsonarray
    # 加载图像并且合并
    image = np.array(img).flatten()

    # 重塑
    image = image.reshape(img_channels, img_rows, img_cols)

    # float32
    image = image.astype('float32')

    # 进行规范化
    image = image / 255

    # 重塑
    rimage = image.reshape(1, img_channels, img_rows, img_cols)

    # 将其提供给NN，获取预测值
    # index = model.predict_classes(rimage)
    # prob_array = model.predict_proba(rimage)

    prob_array = get_output([rimage, 0])[0]
    # print('prob_array: ',prob_array)

    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1

    #最大概率获取输出
    import operator

    guess = max(d.items(), key=operator.itemgetter(1))[0]
    prob = d[guess]

    if prob > 60.0:

        # 将预测保存在json文件中
        # 绘图仪应用程序可以读取以绘制条形图
        # 转存json内容到文件

        jsonarray = d

        return output.index(guess)

    else:
        # 返回索引1的 'Nothing'
        return 1


# %%
def initializers():
    imlist = modlistdir(path2)

    image1 = np.array(Image.open(path2 + '/' + imlist[0]))  # 打开一张照片获取尺寸
    # plt.imshow(im1)

    m, n = image1.shape[0:2]  # 获取图像大小
    total_images = len(imlist)

    # 创建矩阵以存储所有拼合图像
    immatrix = np.array([np.array(Image.open(path2 + '/' + images).convert('L')).flatten()
                         for images in sorted(imlist)], dtype='f')

    print(immatrix.shape)

    input("Press any key")

    # 根据相应的手势类型标记图像集
    label = np.ones((total_images,), dtype=int)

    samples_per_class = int(total_images / nb_classes)
    print("samples_per_class - ", samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class

    '''
    # eg: For 301 img samples/gesture for 4 gesture types
    label[0:301]=0
    label[301:602]=1
    label[602:903]=2
    label[903:]=3
    '''

    data, Label = shuffle(immatrix, label, random_state=2)
    train_data = [data, Label]

    (X, y) = (train_data[0], train_data[1])

    # 分开进行训练和测试

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize
    X_train /= 255
    X_test /= 255

    # 将类向量转换为二元类矩阵
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test


def trainModel(model):
    # 将x和y拆分为训练集和测试集
    X_train, X_test, Y_train, Y_test = initializers()

    # 现在开始训练加载的模型
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                     verbose=1, validation_split=0.2)

    ans = input("Do you want to save the trained weights - y/n ?")
    if ans == 'y':
        filename = input("Enter file name - ")
        fname = path + str(filename) + ".hdf5"
        model.save_weights(fname, overwrite=True)
    else:
        model.save_weights("newWeight.hdf5", overwrite=True)

    visualizeHis(hist)

    # 保存model
    # model.save("newModel.hdf5")


# %%

def visualizeHis(hist):
    # 可视化损耗与准确性
    keylist = hist.history.keys()
    # print(hist.history.keys())
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']


    if 'acc' in keylist:
        train_acc = hist.history['acc']
        val_acc = hist.history['val_acc']
    else:
        train_acc = hist.history['accuracy']
        val_acc = hist.history['val_accuracy']
    xc = range(nb_epoch)

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])


    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)

    plt.show()


# %%
def visualizeLayers(model):
    imlist = modlistdir('./imgs')
    if len(imlist) == 0:
        print('Error: No sample image file found under \'./imgs\' folder.')
        return
    else:
        print('Found these sample image files - {}'.format(imlist))

    img = int(input("Which sample image file to load (enter the INDEX of it, which starts from 0): "))
    layerIndex = int(input("Enter which layer to visualize. Enter -1 to visualize all layers possible: "))

    if img <= len(imlist):

        image = np.array(Image.open('./imgs/' + imlist[img]).convert('L')).flatten()

        ## Predict
        print('Guessed Gesture is {}'.format(output[guessGesture(model, image)]))

        # reshape it
        image = image.reshape(img_channels, img_rows, img_cols)

        # float32
        image = image.astype('float32')

        # normalize it
        image = image / 255

        # reshape for NN
        input_image = image.reshape(1, img_channels, img_rows, img_cols)
    else:
        print('Wrong file index entered !!')
        return

    # 可视化中间层
    # output_layer = model.layers[layerIndex].output
    # output_fn = theano.function([model.layers[0].input], output_layer)
    # output_image = output_fn(input_image)

    if layerIndex >= 1:
        visualizeLayer(model, img, input_image, layerIndex)
    else:
        tlayers = len(model.layers[:])
        print("Total layers - {}".format(tlayers))
        for i in range(1, tlayers):
            visualizeLayer(model, img, input_image, i)


# %%
def visualizeLayer(model, img, input_image, layerIndex):
    layer = model.layers[layerIndex]

    get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])
    activations = get_activations([input_image, 0])[0]
    output_image = activations

    # 如果是四维，取最后一个维度值，因为它不是过滤器
    if output_image.ndim == 4:
        # 重新排列维度，以便于绘制结果
        output_image = np.moveaxis(output_image, 1, 3)

        print("Dumping filter data of layer{} - {}".format(layerIndex, layer.__class__.__name__))
        filters = len(output_image[0, 0, 0, :])

        fig = plt.figure(figsize=(8, 8))
        # 绘制输入图像的32个过滤器数据
        for i in range(filters):
            ax = fig.add_subplot(6, 6, i + 1)

            ax.imshow(output_image[0, :, :, i], 'gray')
            # ax.set_title("Feature map of layer#{} \ncalled '{}' \nof type {} ".format(layerIndex,
            #                layer.name,layer.__class__.__name__))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        plt.tight_layout()
        # plt.show()
        savedfilename = "img_" + str(img) + "_layer" + str(layerIndex) + "_" + layer.__class__.__name__ + ".png"
        fig.savefig(savedfilename)
        print("Create file - {}".format(savedfilename))
        # plt.close(fig)
    else:
        print("Can't dump data of this layer{}- {}".format(layerIndex, layer.__class__.__name__))


