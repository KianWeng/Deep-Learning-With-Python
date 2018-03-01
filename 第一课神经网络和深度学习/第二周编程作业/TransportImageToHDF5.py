import os
import glob
import numpy as np
import scipy.misc as misc
import h5py
import matplotlib.pyplot as plt
import time
import pylab

def readImageAndTransform(fileName, widthPx, heightPx):
    """
    Desc:
    读取图片文件转换成widthPx*heightPx像素图片

    Arguments：
    fileName -- 文件名
    widthPx -- 转换后图片的宽（单位：像素）
    heightPx -- 转换后图片的高（单位：像素）

    Return：
    resizeImageArray -- 转换后的图片
    """
    image = misc.imread(fileName)
    resizeImage = misc.imresize(image, size = (widthPx, heightPx))
    resizeImageArray = np.array(resizeImage)

    return resizeImageArray

def convertFileNameToData(fileName):
    """
    Desc:
    将图片的名字转换成图片内容的属性，并用0,1表示（即0：表示非此类属性图片，1：表示属于此类图片）

    Argument:
    fileName -- 图片名字

    Return：
    data -- 图片的属性值
    """
    dataStr = fileName.split("_")[1].split(".")[0]
    data = int(dataStr)

    return data

def storeDataToHDF5(fileList, widthPx, heightPx, h5FileName):
    """
    Desc:
    将一系列图片转换为widthPx * heightPx像素并存储为HDF5格式的文件

    Arguments:
    fileList -- 文件名集合
    widthPx -- 转换后图片的宽（单位：像素）
    heightPx -- 转换后图片的高（单位：像素）
    h5FileName -- 转换后h5文件的名字
    """
    file = h5py.File(h5FileName, 'w')
    fileListLen = len(fileList)
    trainSetX = np.zeros((fileListLen, widthPx, heightPx, 3))
    trainSetY = np.zeros((fileListLen, 1))
    i = 0
    for fileName in fileList:
        trainSetX[i] = readImageAndTransform(fileName, widthPx, heightPx)
        trainSetY[i] = convertFileNameToData(fileName.split("\\")[-1])
        i += 1
    #print(trainSetX.shape)
    #print(trainSetY.shape)

    file.create_dataset('train_set_x', data = trainSetX)
    file.create_dataset('train_set_y', data = trainSetY)
    file.close

def loadDataset(datasetName):
    """
    Desc:
    导入数据集的数据

    Argument:
    datasetName -- 数据集名字
    """
    dataset = h5py.File(datasetName, 'r')
    datasetXOrig = np.array(dataset["train_set_x"][:])
    datasetYOrig = np.array(dataset["train_set_y"][:])

    #testDataset = h5py.File('datasets/test_catvnoncat.h5', 'r')
    #testSetXOrig = np.array(test_dataset["test_set_x"][:])
    #testSetYOrig = np.array(test_dataset["test_set_y"][:])

    #classes = np.array(test_dataset["list_classes"][:])

    #datasetYOrig = datasetYOrig.reshape((1, datasetYOrig.shape[0]))
    #testSetYOrig = testSetYOrig.reshape((1, test_set_y_orig.shape[0]))

    return datasetXOrig, datasetYOrig

if __name__ == "__main__":
    #imageDir = r"D:\work\Code\Deep-Learning-With-Python\第一课神经网络和深度学习\第二周编程作业\images"
    #fileGlob = os.path.join(imageDir, 'training', '*.' + 'jpg')
    #fileList = []
    #fileList.extend(glob.glob(fileGlob))
    #print(fileList[0].split("\\")[-1])
    #storeDataToHDF5(fileList, 64, 64, 'datasets/train_dogvnondog.h5')    

    #time.sleep(2)

    datasetXOrig, datasetYOrig = loadDataset('datasets/train_dogvnondog.h5')
    print(datasetXOrig.shape)
    print(datasetXOrig[20].shape)
    print(datasetYOrig.shape)
    print(datasetYOrig)
    plt.imshow(datasetXOrig[5])
    #plt.show()
    misc.imsave('2.jpg', datasetXOrig[5])
