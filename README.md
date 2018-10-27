



### 说明：
1、本例使用smo算法实现手写数字0-9识别，训练数据来自MNIST数据集http://yann.lecun.com/exdb/mnist/<br/>
2、代码中部分注释涉及公式，具体可参见https://blog.csdn.net/eeeee123456/article/details/80108096

### 在终端运行的指令：
``` python
# 程序入口为main函数
# main(C, toler, maxIter, kTup, number1, number2)
# C：惩罚参数，toler：精度，maxIter：最大循环次数，kTup：高斯核的带宽，number1：从6万条数据中随机抽取的作为训练集的数目，number2：从6万条数据中随机抽取的作为测试集的数目

import smo
smo.main(100, 0.1, 20, 1, 1000, 200)
```
