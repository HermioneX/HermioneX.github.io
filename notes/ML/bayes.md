#朴素贝叶斯理论

##

#### 1 概述 ####

“朴素”指的是整个过程只做最原始、最简单的假设。</br>
	
######贝叶斯决策理论核心思想：</br>
假设数据集由两类数据组成，数据点（x,y）属于类别1和类别2的概率分别为p1(x,y)和p2(x,y)</br>
>（1）如果p1(x,y)>p2(x,y)，则数据点的类别为1</br>
>（2）如果p1(x,y)<p2(x,y)，则数据点的类别为2</br>

即选择概率对应更高的类别

######条件概率：</br>
简单的条件概率公式可以表示为：p(c|x)=p(x|c)p(c)/p(x)</br>
应用于分类器中，即是通过以上公式来计算数据点分别来源于类别1和2的概率，分别用p(c1|x,y)和p(c2|x,y)表示,条件与结果概率可以通过上面给出的条件概率公式换算。</br>

因此，贝叶斯分类准则为：</br>
>（1）如果p(c1|x,y)>p(c2|x,y)，则数据点的类别为1</br>
>（2）如果p(c1|x,y)<p(c2|x,y)，则数据点的类别为2</br>


使用朴素贝叶斯进行文档分类的一般流程：
> 　　（１）收集数据：任何方法</br>
> 　　（２）准备数据：需要数值型或者布尔型数据</br>
> 　　（３）分析数据：大量特征时，使用直方图绘制特征</br>
> 　　（４）训练算法：计算不独立特征的条件概率</br>
> 　　（５）测试算法：计算错误率</br>
> 　　（６）使用算法：常见应用是文档分类，可以用于任意分类场景。
> 　　
#### 2  使用Python进行文本分类####

示例：以在线社区的留言板为例，需要屏蔽侮辱性的言论，因此需要一个快速过滤器来进行筛选。
#### 2.1  准备数据：从文本中构建向量####

建立bayes.py文件，创建DataSet


    def loadDataSet():    
	#创建一些实验样本
    postingList=[['my','dog','has','flea',\
                  'problem','help','please'],
                 ['maybe','not','take','him',\
                  'to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute',\
                  'I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how',\
                  'to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec
	#返回词条切分后的文档集合和标签集合
	#1和0分别表示侮辱性和非侮辱性文档，由人工标注，用于训练程序自动检测侮辱性留言

	def createVocabList(dataSet):     
    vocabSet = set([])
	#创建一个空集
    for document in dataSet:
        vocabSet = vocabSet|set(document)
		#将每篇文档返回的新词添加到集合中，形成新的并集
    return list(vocabSet)
	#返回一个包含在所有文档中出现的不重复词的列表

	def setOfWords2Vec(vocabList, inputSet):         #get situation
    returnVec = [0]*len(vocabList)
	#创建一个长度等于词表长度，值为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
			#如果输入的词在词表中，则向量对应位置为1
        else: print "the word :%s is not in my Vocabulary!" %word
    return returnVec
	#返回一个向量，用0，1定位输入词在词表中的位置

#### 2.2  训练算法：从词向量计算概率####

伪代码如下：
	
	计算每个类别中的文档数目
	对每篇训练文档：
		对每个类别：
			如果词条出现在文档中->增加该词条的计数值
			增加所有词条的计数值
	对每个类别：
		对每个词条：
			将每个类别词条的数目除以该类别总词条数目得到条件概率
	返回每个类别的条件概率

将from numpy import * 等语句添加到bayes.py文件之前，并添加以下程序：

	def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)   
	#训练文档中侮辱文档数/总的训练文档数=侮辱文档的概率
    p0Num = zeros(numWords);p1Num = zeros(numWords)
	#初始化概率向量，分别表示每个词对应出现的概率
    p0Denom = 0.0; p1Denom =0.0
    for i in range(numTrainDocs):
	#对于每一篇文档
        if trainCategory[i] == 1:
		#若为侮辱性文档，则该p1向量加该文档向量，总侮辱性词汇数量加该文档词总数
		#第二种理解方法：若某一词在某文档中出现，则根据文档类别在相应的文档向量中加1，在总文档中数目也加1
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
	#侮辱性词汇向量/总侮辱性词汇总量=每个侮辱性词汇占总侮辱性词汇概率
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive

#### 2.3  测试算法：根据实际情况修改分类器####

如果对于某个类别，有n个特征值，则计算文档属于某个类别i的概率需要将多个概率乘积相乘，即：p(w0|ci)p(w1|ci)p(w2|ci).....p(wn|ci),若存在概率值为0的情况，则总的概率值即为0.为避免这种情况，将所有词的出现次数初始化为1，分母初始化为2</br>
	
	p0Num = ones(numWords);p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom =2.0

另一个问题为**下溢出**，即由太多很小的数字相乘造成的，解决办法为对乘积取自然对数，由公式：</br>

	ln(a*b) = ln(a)+ln(b)

在分类器中作出对应修改：

	p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)

下面开始构建完整的分类器：

	def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	#输入需要分类的向量和已经通过trainNB0计算得到的三个概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	#根据之前的说明，这里的相加即为元素的相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
	#返回值判别是否为侮辱性文档


	def testingNB():
	#此函数为便利函数，即封装所有操作来节省指令的输入
    listOPosts,listClasses = loadDataSet()
	#创建文档集和类别标签
    myVocabList= createVocabList(listOPosts)
	#生成不重复词的列表
    trainMat=[]
	#创建训练矩阵
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
		#在训练矩阵中添加表明各个文档中对应词在列表中位置的向量
    p0V,p1V,pAb = trainNB0(trainMat,listClasses)
	#返回两种类别各自的词语的条件概率向量和两类别文档各自概率
    testEntry = ['love','my','dalmation']
	#测试文档1
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	#生成文档1中各词汇的位置向量
    print testEntry,'Classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
	#返回对该文档类别的判断结果
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'Classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
	#文档2同理

#### 3  使用朴素贝叶斯过滤垃圾邮件####

之前的工作中，我们对一个词是否出现作为一个特征，这可以描述为**词集模型**（set of words model）。

如果一个词在文档中出现不止一次，则该词可能包含着更多信息，称为**词袋模型**（bag of words model）。

词袋中每个词可以出现多次，为适应词袋模型，对函数setOfWords2Vec作出修改。当遇到一个单词时，会增加词向量中的对应值而不是将其设为1.

	def bagOfWords2Vec(vocabList, inputSet):         
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
使用朴素贝叶斯过滤垃圾邮件的主要流程：
> 　　（１）收集数据：提供文本文件</br>
> 　　（２）准备数据：文本文件解析为词条向量</br>
> 　　（３）分析数据：检查词条确保解析的正确性</br>
> 　　（４）训练算法：使用之前建立的trainNB0（）函数</br>
> 　　（５）测试算法：使用classifyNB（）并构建新的测试函数计算错误率</br>
> 　　（６）使用算法：构建完整程序对一组文档进行分类，将分类错误的输出到屏幕。

#### 2.1  准备数据：切分文本####

首先使用Python的string.split()方法进行划分，但标点符号也被当作词的一部分，可以通过正则表达式来切分句子，其中分隔符是除了单词、数字以外的任意字符串。

	import re
	regEx = re.compile（'\\w*'）
	listOfTokens = regEx.split(mySent)
	
此时获得的词表中包含空的字符串，通过计算字符串长度筛选，同时加入lower指令统一为小写。

	[tok.lower() for tok in listOfTokens if len(tok) > 0]

**PS:在对HTML和URL对象进行解析时，需要更高级的过滤器来进行处理**

#### 2.2  测试算法：使用朴素贝叶斯进行交叉验证####

将以下代码加入bayes.py文件中

**PS：这里额外说明一下Python中一个小的语法：append和extend的区别**</br>
append()向列表尾部追加一个新元素，列表只占一个索引位，在原有列表上增加</br>
extend()向列表尾部追加一个列表，列表中的每个元素都追加进来，在原有列表上增加</br>

	def textParse(bigString):
	#实现对词条的初步切分
    import re
    listOfTokens = re.split(r'\w*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

	def spamTest():
    docList=[]; classList = []; fullText = []
    for i in range(1,26):
	#每个文件夹下各有26个文件
        wordList = textParse(open('E:/Python_code/machinelearninginaction/Ch04/email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
		#这里注意appen和exten的区别
		#doc获得一个新元素，fulltext增加了一个wordlist的元素数
        classList.append(1)
		#读入垃圾邮件，并在类别向量中加入一个1
        wordList = textParse(open('E:/Python_code/machinelearninginaction/Ch04/email/ham/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
		#读入垃圾邮件，并在类别向量中加入一个元素
	#循环结束将得到doclist元素为各个邮件词条列表；fulltext元素为列表所有邮件的词条
    vocabList = createVocabList(docList)
	#创建所有邮件的词表
    trainingSet = range(50); testSet=[]
	#构建一个测试集和一个训练集，所有的50封电子邮件一开始被加入训练集
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
		#随机选取其中的10封作为测试集，注意这里返回值均为0~49的整数
        del(trainingSet[randIndex])
		#将选为测试集的10封邮件从训练集中剔除，此为 留存交叉验证
    trainMat=[]; trainClasses = []
	#初始化测试矩阵和测试标签
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
		#构建训练邮件的词条向量，加入测试矩阵
        trainClasses.append(classList[docIndex]) 
		#构建训练邮件的标签列表
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
	#计算测试邮件中每一个词的条件概率
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
		#创建每一封测试邮件的词向量
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
		#使用classifyNB()判断，与标签向量对应值对比，计算错误率
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ' , float(errorCount)/len(testSet)

#### 3  本章小结####

对于分类而言，使用概率有时比使用硬性规则更加有效。贝叶斯概率及贝叶斯准则提供了一种用已知值估算未知概率的有效方法。本章主要基于条件独立的假设，这个假设较为简单，因此被称为朴素贝叶斯理论。

