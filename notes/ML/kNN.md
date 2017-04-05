#k-临近算法

##

#### 1 概述 ####
　　

简单的说，K-近邻值法采用测量不同特征值之间的距离方法进行分类
　
> 　　优点：精度高，对异常值不敏感，无数据输入假定。</br>
　　缺点：计算复杂度高、空间复杂度高</br>
　　适用范围：数值型和标称型</br>

k-近邻值法的一般流程：
> 　　（１）收集数据：任何方法</br>
> 　　（２）准备数据：计算机里所需要的值，最好是结构化数据</br>
> 　　（３）分析数据：任何方法</br>
> 　　（４）训练算法：不适用</br>
> 　　（５）测试算法：计算错误率</br>
> 　　（６）使用算法：输入样本数据和结构化的输出结果，运行K-临近值判定输入数据的分类，最后对分类执行后续处理。

#### 2　k-近邻算法概述 ####
#### 2.1  使用Python导入数据####
</br>
建立kNN.py文件，导入模块：科学计算包NumPy和运算符模块operator，创建一个DataSet</br>
导入matplotlib包是为了之后的分析
　

    from numpy import *
	import operator
	import matplotlib
	import matplotlib.pyplot as plt
	import os


	def creatDataSet():
    	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    	labels = ['A','A','B','B']
    	return group, labels

group里每组数据包含了两个属性或者特征值，向量label则包含了每个数据的标签信息

#### 2.2 实施kNN算法 ####
</br>
主要是使用k-近邻值算法将每组数据划分到某个类中</br>


	def classify0(inX, dataSet, labels, k):
		#四个输入参数：用于分类的输入向量inX，输入的训练样本集dataSet，标签向量label，选择最近邻居的数目k
    	dataSetSize = dataSet.shape[0]
    	diffMat = tile(inX, (dataSetSize,1)) - dataSet
		#tile函数用inX减去dataSet中的数，获得向量差
    	sqDiffMat = diffMat**2 
    	sqDistances = sqDiffMat.sum(axis=1)
    	distances = sqDistances**0.5 #使用欧式距离公式
    	sortedDistIndicies = distances.argsort()
    	classCount={}
    	for i in range(k):
        	voteIlabel = labels[sortedDistIndicies[i]]
        	classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
		#sorted函数排序，获得从小到大排序结果
    	return sortedClassCount[0][0] #获得最小值

#### 3 示例：使用k-近邻算法 ####

#### 3.1在约会网站上使用k-近邻算法： ####
> 　　（１）收集数据：提供文本文件</br>
> 　　（２）准备数据：使用Python解析文本文件</br>
> 　　（３）分析数据：使用Maiplotlib画二位扩散图</br>
> 　　（４）训练算法：不适用</br>
> 　　（５）测试算法：使用某用户提供的部分数据</br>
> 　　（６）使用算法：产生简单的命令行程序，可以输入特征数据以判断是否喜欢。

将文本记录转换为NumPy的解析程序：

	def file2matrix(filename):
    	fr = open(filename)
    	arrayOLines = fr.readlines() #返回一个包含行的列表
    	numberOfLines = len(arrayOLines) #得到文件行数
    	returnMat = zeros((numberOfLines,3 )) #创建返回的NumPy矩阵，初始为0
    	classLabelVector = []
    	index = 0
    	for line in arrayOLines:
        	line =  line.strip() #截取所有回车字符
        	listFromLine = line.split('\t')  #使用\t分割行数据为元素列表
        	returnMat[index,:] = listFromLine[0:3] #截取每行前三个放入矩阵
        	classLabelVector.append(int(listFromLine[-1]))
        	index +=1
    	return returnMat, classLabelVector

	datingDataMat, datingLabels = file2matrix('E:\Python_code\machinelearninginaction\Ch02\datingTestSet2.txt')

#### 3.2 分析数据：使用Matplotlib 画图 ####
 
	#print datingDataMat

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.scatter(datingDataMat[:,1], datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
	# 
	# plt.show()

获得了散点图，绘制了色彩不等、尺寸不同的点

#### 3.3 准备数据：归一化数值 ####

　　各个样本的量级不同，使用欧式距离公式使得量级大的特征影响远远大于其他，为了消除这个影响，应当进行归一化处理

主要公式为：</br>
　　newValue = (oldValue-min)/(max-min)

	def autoNorm(dataSet):
    	minVals = dataSet.min(0)
    	maxVals = dataSet.max(0)
    	ranges = maxVals - minVals
    	normDataSet = zeros(shape(dataSet))
    	m = dataSet.shape[0]
    	normDataSet = dataSet - tile(minVals, (m,1))
    	normDataSet = normDataSet/tile(ranges, (m,1))
    	return normDataSet, ranges, minVals 
    
	normMat, ranges, minVals = autoNorm(datingDataMat)    

#### 3.4 测试算法：作为完整程序验证分类器 ####

　　本节来测试分类器的效果。通常提供已有数据的90%来训练样本来训练分类器，使用其余的10%来测试分类器，检测分类器的正确率。注意10%的数据应该是随机选择的
    
	def datingClassTest():
    	hoRatio = 0.10      #使用10%的数据
    	datingDataMat,datingLabels = file2matrix('E:\Python_code\machinelearninginaction\Ch02\datingTestSet2.txt')       
    	normMat, ranges, minVals = autoNorm(datingDataMat) #归一化
    	m = normMat.shape[0]
    	numTestVecs = int(m*hoRatio) #计算测试向量数量
    	errorCount = 0.0
    	for i in range(numTestVecs): #使用分类器进行测试
        	classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        	print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        	if (classifierResult != datingLabels[i]): errorCount += 1.0
			#与label对比，计算错误率
    	print "the total error rate is: %f" % (errorCount/float(numTestVecs))
        
	#datingClassTest()        


#### 4 手写识别系统   </br>

　　本节构造使用k-近邻算法的手写识别系统。此处只能识别数字0-9，为了便于理解将图像转换为文本格式。

> 　　（１）收集数据：提供文本文件</br>
> 　　（２）准备数据：编写函数img2vector，图像格式转换为向量格式</br>
> 　　（３）分析数据：在Python命令符中检查数据</br>
> 　　（４）训练算法：不适用</br>
> 　　（５）测试算法：提供部分数据集为测试样本</br>
> 　　（６）使用算法：无。
> 　　
#### 4.1 准备数据：将图像转换为测试向量 ####
　把一个32\*32的二进制图像矩阵转化为1*1024的向量，然后循环读出文件的前32行，并存储每行的前32个字符值，最后返回数组。


	def img2vector(filename):
    	returnVect = zeros((1,1024))
    	fr = open(filename)
    	for i in range(32):
        	lineStr = fr.readline()
        	for j in range(32):
            	returnVect[0,32*i+j] = int(lineStr[j])
    	return returnVect

	testVector = img2vector(r'E:\Python_code\machinelearninginaction\Ch02\0_13.txt')

	#print testVector[0,0:31]

#### 4.2 测试算法：用k-近邻算法识别手写数字 ####

	def handwritingClassTest():
    	hwLabels = []
    	trainingFileList = os.listdir(r'E:\Python_code\machinelearninginaction\Ch02\trainingDigits')  
		#获取目录内容
    	m = len(trainingFileList)
    	trainingMat = zeros((m,1024))
    	for i in range(m):
        	fileNameStr = trainingFileList[i]
        	fileStr = fileNameStr.split('.')[0]     
        	classNumStr = int(fileStr.split('_')[0]) #通过文件名解析数字
        	hwLabels.append(classNumStr) #存入标签
        	trainingMat[i,:] = img2vector(r'E:\Python_code\machinelearninginaction\Ch02\trainingDigits\%s' % fileNameStr)
    	testFileList = os.listdir(r'E:\Python_code\machinelearninginaction\Ch02\testDigits')        #遍历文件名
    	errorCount = 0.0
    	mTest = len(testFileList)
    	for i in range(mTest):
        	fileNameStr = testFileList[i]
        	print fileNameStr
        	fileStr = fileNameStr.split('.')[0]     
        	classNumStr = int(fileStr.split('_')[0])
        	vectorUnderTest = img2vector(r'E:\Python_code\machinelearninginaction\Ch02\testDigits\%s' % fileNameStr)
        	classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        	print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        	if (classifierResult != classNumStr): errorCount += 1.0
    	print "\nthe total number of errors is: %d" % errorCount
    	print "\nthe total error rate is: %f" % (errorCount/float(mTest))


    
	handwritingClassTest()    

#### 5 本章小结 ####

　　k-近邻算法是分类数据最简单最有效的方法，本章通过两个例子讲述了如何使用k-近邻算法