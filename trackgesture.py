# -*- coding: utf-8 -*-
"""
Created on Mon May 13 01:01:43 2024

author: leidaqian
"""
# %%
import cv2
import numpy as np
import os
import time

import threading

import gestureCNN as myNN

minValue = 70

x0 = 400
y0 = 200
height = 200
width = 200

saveImg = False
guessGesture = False
visualize = False

kernel = np.ones((15, 15), np.uint8)
kernel2 = np.ones((1, 1), np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 使用掩码模式skinmask
binaryMode = True
bkgrndSubMode = False
mask = 0
bkgrnd = 0
counter = 0
# 此参数控制要按手势拍摄的图像样本数
numOfSamples = 301
gestname = ""
path = ""
mod = 0

banner = '''\nWhat would you like to do ?
    1- Use pretrained model for gesture recognition & layer visualization
    2- Train the model (you will require image samples for training under .\imgfolder)
    3- Visualize feature maps of different layers of trained model
    4- Exit	
    '''


# %%
def saveROIImg(img):
    global counter, gestname, path, saveImg
    if counter > (numOfSamples - 1):
        # Reset the parameters
        saveImg = False
        gestname = ''
        counter = 0
        return

    counter = counter + 1
    name = gestname + str(counter)
    print("Saving img:", name)
    cv2.imwrite(path + name + ".png", img)
    time.sleep(0.04)


# %%
def skinMask(frame, x0, y0, width, height, framecount, plot):
    global guessGesture, visualize, mod, saveImg
    # HSV值
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])

    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)

    roi = frame[y0:y0 + height, x0:x0 + width]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 应用肤色系列
    mask = cv2.inRange(hsv, low_range, upper_range)

    mask = cv2.erode(mask, skinkernel, iterations=1)
    mask = cv2.dilate(mask, skinkernel, iterations=1)

    # blur
    mask = cv2.GaussianBlur(mask, (15, 15), 1)

    # 按位和遮罩原始帧
    res = cv2.bitwise_and(roi, roi, mask=mask)
    # 颜色到灰度
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and (framecount % 5) == 4:

        t = threading.Thread(target=myNN.guessGesture, args=[mod, res])
        t.start()
    elif visualize == True:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(0)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False

    return res


# %%
def binaryMask(frame, x0, y0, width, height, framecount, plot):
    global guessGesture, visualize, mod, saveImg

    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
    # roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = frame[y0:y0 + height, x0:x0 + width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and (framecount % 5) == 4:
        # ores = cv2.UMat.get(res)
        t = threading.Thread(target=myNN.guessGesture, args=[mod, res])
        t.start()
    elif visualize == True:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(1)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False

    return res



def bkgrndSubMask(frame, x0, y0, width, height, framecount, plot):
    global guessGesture, takebkgrndSubMask, visualize, mod, bkgrnd, saveImg

    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 1)
    roi = frame[y0:y0 + height, x0:x0 + width]
    # roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Take background image
    if takebkgrndSubMask == True:
        bkgrnd = roi
        takebkgrndSubMask = False
        print("Refreshing background image for mask...")

    # 在roi和bkgrnd图像内容之间进行比较
    diff = cv2.absdiff(roi, bkgrnd)

    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    mask = cv2.GaussianBlur(diff, (3, 3), 5)
    mask = cv2.erode(diff, skinkernel, iterations=1)
    mask = cv2.dilate(diff, skinkernel, iterations=1)
    res = cv2.bitwise_and(roi, roi, mask=mask)

    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and (framecount % 5) == 4:
        t = threading.Thread(target=myNN.guessGesture, args=[mod, res])
        t.start()
        # t.join()
        # myNN.update(plot)

    elif visualize == True:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(0)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False

    return res


# %%
def Main():
    global guessGesture, visualize, mod, binaryMode, bkgrndSubMode, mask, takebkgrndSubMask, x0, y0, width, height, saveImg, gestname, path
    quietMode = False

    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    fx = 10
    fy = 350
    fh = 18

    #调用cnn模型加载回调
    while True:
        ans = int(input(banner))
        if ans == 1:
            mod = myNN.loadCNN()
            break
        elif ans == 2:
            mod = myNN.loadCNN(True)
            myNN.trainModel(mod)
            input("Press any key to continue")
            break
        elif ans == 3:
            if not mod:
                mod = myNN.loadCNN()
            else:
                print("Will load default weight file")

            myNN.visualizeLayers(mod)
            input("Press any key to continue")
            continue

        else:
            print("Get out of here!!!")
            return 0

    #抓取相机输入
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    # set rt size as 640x480
    ret = cap.set(3, 640)
    ret = cap.set(4, 480)

    framecount = 0
    fps = ""
    start = time.time()

    plot = np.zeros((512, 512, 3), np.uint8)

    while (True):
        ret, frame = cap.read()
        max_area = 0

        frame = cv2.flip(frame, 3)
        frame = cv2.resize(frame, (640, 480))

        if ret == True:
            if bkgrndSubMode == True:
                roi = bkgrndSubMask(frame, x0, y0, width, height, framecount, plot)
            elif binaryMode == True:
                roi = binaryMask(frame, x0, y0, width, height, framecount, plot)
            else:
                roi = skinMask(frame, x0, y0, width, height, framecount, plot)

            framecount = framecount + 1
            end = time.time()
            timediff = (end - start)
            if (timediff >= 1):
                # timediff = end - start
                fps = 'FPS:%s' % (framecount)
                start = time.time()
                framecount = 0

        cv2.putText(frame, fps, (10, 20), font, 0.7, (0, 255, 0), 2, 1)
        cv2.putText(frame, 'Options:', (fx, fy), font, 0.7, (0, 255, 0), 2, 1)
        cv2.putText(frame, 'b - Toggle Binary/SkinMask', (fx, fy + fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, 'x - Toggle Background Sub Mask', (fx, fy + 2 * fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, 'g - Toggle Prediction Mode', (fx, fy + 3 * fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, 'q - Toggle Quiet Mode', (fx, fy + 4 * fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, 'n - To enter name of new gesture folder', (fx, fy + 5 * fh), font, size, (0, 255, 0), 1, 1)
        cv2.putText(frame, 's - To start capturing new gestures for training', (fx, fy + 6 * fh), font, size,
                    (0, 255, 0), 1, 1)
        cv2.putText(frame, 'ESC - Exit', (fx, fy + 7 * fh), font, size, (0, 255, 0), 1, 1)

        # 如果启用，将停止更新主OpenCV窗口
        #降低一些处理能力
        if not quietMode:
            cv2.imshow('Original', frame)
            cv2.imshow('ROI', roi)

            if guessGesture == True:
                plot = np.zeros((512, 512, 3), np.uint8)
                plot = myNN.update(plot)

            cv2.imshow('Gesture Probability', plot)
            # plot = np.zeros((512,512,3), np.uint8)

        key = cv2.waitKey(5) & 0xff

        # 按esc键关闭
        if key == 27:
            break

        #  使用 b 键在二进制阈值或基于皮肤掩码的过滤器之间切换
        elif key == ord('b'):
            binaryMode = not binaryMode
            bkgrndSubMode = False
            if binaryMode:
                print("Binary Threshold filter active")
            else:
                print("SkinMask filter active")

        # 使用 x 键使用和刷新背景子掩码滤镜
        elif key == ord('x'):
            takebkgrndSubMask = True
            bkgrndSubMode = True
            print("BkgrndSubMask filter active")


        # 使用 g 键通过 CNN 开始手势预测
        elif key == ord('g'):
            guessGesture = not guessGesture
            print("Prediction Mode - {}".format(guessGesture))

        #使用v键可视化图层
        # elif key == ord('v'):
        #    visualize = True

        # 使用 i，j，k，l 调整 ROI 窗口
        elif key == ord('i'):
            y0 = y0 - 5
        elif key == ord('k'):
            y0 = y0 + 5
        elif key == ord('j'):
            x0 = x0 - 5
        elif key == ord('l'):
            x0 = x0 + 5

        # 隐藏手势窗口的静音模式
        elif key == ord('q'):
            quietMode = not quietMode
            print("Quiet Mode - {}".format(quietMode))

        # 使用 s 键开始/暂停/恢复拍摄快照
        # numOfSamples 控制要拍摄的快照数量 PER 手势
        elif key == ord('s'):
            saveImg = not saveImg

            if gestname != '':
                saveImg = True
            else:
                print("Enter a gesture group name first, by pressing 'n'")
                saveImg = False

        # 使用n输入手势名称
        elif key == ord('n'):
            gestname = input("Enter the gesture folder name: ")
            try:
                os.makedirs(gestname)
            except OSError as e:
                # if directory already present
                if e.errno != 17:
                    print('Some issue while creating the directory named -' + gestname)

            path = "./" + gestname + "/"

        # elif key != 255:
        #    print key

    # Realse & destroy
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    Main()