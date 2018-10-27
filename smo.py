# -- coding: utf-8 --
from numpy import *
import numpy as np
import random
import datetime

class optStruct:
    def __init__(self, trainData, trainLabel, C, toler, kTup):
        # 样本X, 样本Y, 惩罚参数C, 精度e, 核函数
        self.X = trainData
        self.labelMat = trainLabel
        self.C = C
        self.tol = toler
        self.m = shape(trainData)[0]
        self.alphas = mat(zeros((self.m,1))) #初始化alphas为0
        self.b = 0                           #初始化b为0
        self.eCache = mat(zeros((self.m,2))) #初始化Ek为0
        self.K = mat(zeros((self.m,self.m)))
        rbf = -kTup**2
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], self.m, rbf)

class dataSet:
    def __init__(self, r1, r2, number1, number2):
        self.dataFile = 'data'
        self.labelFile = 'labels'
        self.r1 = r1
        self.r2 = r2
        self.number1 = number1
        self.number2 = number2

# 主函数，程序入口
def main(C, toler, maxIter, kTup, number1, number2):
    # 惩罚参数C, 精度e, 最大循环次数maxIter, 核, 训练集数量, 测试集数量
    starttime = datetime.datetime.now()
    data, label = loadSet(number1, number2)
    trainData, testData = data
    trainLabel, testLabel, trainLabelForTest = label
    x, y, m = shape(trainData)[0], shape(trainLabel)[1], shape(testData)[0]
    trainResult, testResult = zeros((x, y)), zeros((m, y))
    labelMat = mat(trainLabel)
    oS = optStruct(mat(trainData), mat([]), C, toler, kTup)
    endtime1 = datetime.datetime.now()
    print '核函数运行时间：', (endtime1-starttime).seconds, 's'
    for i in range(y):
        label = array(labelMat[:,i])
        oS.labelMat = mat(label)
        oS.alphas = mat(zeros((x,1)))
        oS.b = 0
        oS.eCache = mat(zeros((x,2)))
        alphas, b = smo(oS, C, maxIter)
        w = calcWs(alphas, trainData, label)
        for index in range(x):
            trainResult[:, i][index] = trainData[index]*mat(w) + b
        for index in range(m):
            testResult[:, i][index] = testData[index]*mat(w) + b
        endtime3 = datetime.datetime.now()
        print i, (endtime3-starttime).seconds, 's'
    evaluate('训练集准确率', number1, trainResult, trainLabelForTest)
    evaluate('测试集准确率', number2, testResult, testLabel)
    endtime2 = datetime.datetime.now()
    print '总运行时间：', (endtime2-starttime).seconds, 's'

def loadSet(number1, number2):
    # number1为训练集数量，number2为测试集数量
    r1 = random.randint(1,60000 - number1)
    r2 = random.randint(1,60000 - number2)
    DS = dataSet(r1, r2, number1, number2)
    return loadDataSet(DS), loadLabelSet(DS)

def loadDataSet(DS):
    # 图片有784个元素（28×28），将其转化成784个特征，每个特征为一个10进制数
    # 每行16个元素，49行构成一张图片
    numTR = (DS.r1 - 1) * 49
    numTE = (DS.r2 - 1) * 49
    trainData, testData, trainLine, testLine = [], [], [], []
    trainStart, trainEnd, testStart, testEnd = False, False, False, False
    fr = open(DS.dataFile)
    for index, line in enumerate(fr.readlines()):
        if (index == numTR): trainStart = True
        if (index == numTR + 49 * DS.number1): trainEnd = True
        if (index == numTE): testStart = True
        if (index == numTE + 49 * DS.number2): testEnd = True

        if (trainStart & (not trainEnd)):
            lists = list(line.replace(' ', '').replace('\n', ''))
            for i in range(len(lists) / 2):
                trainLine.append(int(lists[i*2] + lists[(i*2)+1], 16))
            if ((index + 1) % 49 == 0):
                trainData.append(trainLine)
                trainLine = []

        if (testStart & (not testEnd)):
            lists = list(line.replace(' ', '').replace('\n', ''))
            for i in range(len(lists) / 2):
                testLine.append(int(lists[i*2] + lists[(i*2)+1], 16))
            if ((index + 1) % 49 == 0):
                testData.append(testLine)
                testLine = []

        if (trainEnd & testEnd): break
    return trainData, testData

def loadLabelSet(DS):
    numTR = (DS.r1 - 1) * 2
    numTE = (DS.r2 - 1) * 2
    trainLabel = -ones((DS.number1, 10))
    trainLabelForTest, testLabel = [], []
    allLine = ''
    fr = open(DS.labelFile)
    for line in fr.readlines():
        allLine += line.replace(' ', '')
    lists = list(allLine.replace('\n', ''))
    for i in range(DS.number1):
        num1 = int(lists[numTR + i * 2] + lists[numTR + i * 2 + 1], 16)
        trainLabel[i, :][num1] = 1
        trainLabelForTest.append(num1)
    for i in range(DS.number2):
        num2 = int(lists[numTE + i * 2] + lists[numTE + i * 2 + 1], 16)
        testLabel.append(num2)
    return trainLabel, testLabel, trainLabelForTest

def kernelTrans(X, A, m, rbf):
    # 定义核函数，计算K(x,xi)
    # X为数据集，A为数据集的一行
    deltaRow = X - A.repeat(m, axis=0)
    K = multiply(deltaRow, deltaRow).sum(axis=1)
    K = exp(K/rbf)
    return K

def smo(oS, C, maxIter):
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        # 循环次数大于maxIter 或 alphaPairsChanged连续2次等于0，则退出循环
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):                    # 循环所有alphas，作为smo的第一个变量
                alphaPairsChanged += inner(i,oS)     # inner函数：如果有任意一对alphas值发生改变，返回1
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:                     # 循环大于0小于C的alphas，作为smo的第一个变量
                alphaPairsChanged += inner(i,oS)
            iter += 1
        if entireSet: entireSet = False
        elif (alphaPairsChanged == 0): entireSet = True
    return oS.alphas, oS.b

def inner(i, oS):
    Ei = calcEk(oS, i)                                              #计算E1
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
    # (yiEi<0 => yig(xi)<1 and alpha<C) or (yiEi>0 => yig(xi)>1 and alpha>0) 选取违背K-T条件的alpha1
        j,Ej = selectJ(i, oS, Ei)                                   # 选择smo的第二个变量
        alphaIold = oS.alphas[i].copy()                             # alpha1old
        alphaJold = oS.alphas[j].copy()                             # alpha2old
        if (oS.labelMat[i] != oS.labelMat[j]):                      # y1、y2异号
            L = max(0, oS.alphas[j] - oS.alphas[i])                 # L=max(0,alpha2old-alpha1old)
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])       # H=min(0,C+alpha2old-alpha1old)
        else:                                                       # y1、y2同号
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)          # L=max(0,alpha2old+alpha1old-C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])              # H=min(0,alpha2old+alpha1old)
        if L==H: return 0                                           # 异常情况，返回
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]               # 2K12-K11-K22
        if eta >= 0: return 0                                       # 异常情况，返回
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta                # 式(35)计算alpha2new,unc
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)                  # 返回alpha2new
        updateEk(oS, j)                                             # 更新E2的值
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): return 0      # alpha2基本不变，返回
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])      #式(36)求alpha1new
        updateEk(oS, i)                                             #更新E1的值
        #b1new
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        #b2new
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        #根据b值的计算结论更新b值
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def calcEk(oS, k):
    # 该函数用于计算Ek
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])                                 # 对应式(34)Ei=g(xi)-yi
    return Ek

def selectJ(i, oS, Ei):
    #返回使｜E1-E2｜值最大的E2及E2的位置j
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]                          # 更新位置i的值E1，用1标示该值为更新过的Ei
    validEcacheList = nonzero(oS.eCache[:,0].A)[0] # 返回eCache数组第一列不为0的值，即更新过的Ei的索引
    if (len(validEcacheList)) > 1:
        # 如果eCache的更新过的Ek个数大于1，返回使｜E1-E2｜值最大的E2及E2的位置j
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:
        # 如果eCache的Ek都是没有更新过的，则随机选择一个与E1不同位置的E2并返回E2的位置j
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def selectJrand(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

def calcWs(alphas,dataArr,trainLabel):
    # 利用式(25)alphas[i]*y[i]*xi求w的值
    X = mat(dataArr)
    labelMat = trainLabel
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def evaluate(types, number, result, label):
    res = []
    for i in range(len(result)):
        res.append((np.argmax(result[i]), label[i]))
    print types, '：', sum(int(x == y) for (x, y) in res), '/', number

