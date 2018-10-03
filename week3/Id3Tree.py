'''
id3算法：

首先，ID3算法需要解决的问题是如何选择特征作为划分数据集的标准。在ID3算法中，选择信息增益最大的属性作为当前的特征对数据集分类。
    信息增益的概念将在下面介绍，通过不断的选择特征对数据集不断划分；  
其次，ID3算法需要解决的问题是如何判断划分的结束。分为两种情况，
    第一种为划分出来的类属于同一个类，如上图中的最左端的“非鱼类”，即为数据集中的第5行和第6行数据；最右边的“鱼类”，即为数据集中的第2行和第3行数据。
    第二种为已经没有属性可供再分了。此时就结束了。
通过迭代的方式，我们就可以得到这样的决策树模型。



'''

import math;

class Tree:
    def __init__(self,parent = None):
        self.parent = parent
        self.children = []
        self.splitFeature = None
        self.splitFeatureValue = None
        self.label = None


def dataToDistribution(data):
    '''
    根据class 来计算distribution
    Turn a dataset which has n possible callsfication labels into a probability distribution with n entries
    :param data:  features,labels
    :return: distriburibution for dataset
    '''

    ...
    allLabels  = [label for (point,label) in data]
    numEntries = len(allLabels)
    posibleLabels = set(allLabels)

    dist = []
    for aLabels in posibleLabels:
        dist.append(float(allLabels.count(aLabels))/numEntries)
    return dist;

def entropy(dist):
    '''

    :param dist: distribution for n kinds of classification
    :return: entropy of this data
    '''

    return -sum(p* math.log(p,2) for p in dist);


def splitData(data,featureIndex):
    '''
    :param data:
    :param featureIndex:
    :return:dataSubset
    '''
    # 所有数据中选择的划分属性的value
    attrValues = [point[featureIndex] for (point,label) in data]
    # 对该属性以不同值划分
    for aValue in set(attrValues):
        dataSubset = [(point ,label) for (point,label) in  data if point[featureIndex] == aValue]

        yield dataSubset;

def gain(data,featureIndex):
    '''
    compute the expected gain(信息增益) from splitting the data dalong all possible valuies of  feature
    :param data:
    :param featureIndex:
    :return:
    '''
    entropyGain = entropy(dataToDistribution(data))
    for dataSubset  in splitData(data,featureIndex):
        entropyGain -= entropy(dataToDistribution(dataSubset))

    return entropyGain
def homogeneous(data):
    ''' Return true if  all the data have same label,and False otherwise'''
    return len(set([label for (point ,label )in data])) <=1

def majorityVote(data,node):
    '''多数投票解决特征一样而分类不一样的数据
    Label node with the majority of the classes label in the given data set
    :param data:
    :param node:
    :return:
    '''
    #先拿出所有标签
    labels = [label for (pt,label) in data]
    choice = max (set(labels), key = labels.count)
    node.label = choice
    return node

def buildDecisionTree(data,root,remainingFeatures):
    '''Build a Decisiion treee from the given data,appending the children to the given root node (whick may be the root of a subtree'''
    #全都是一类
    if homogeneous(data):
        root.label = data[0][1]
        return root
    #剩下的features都一样，但是class 不一样
    if len(remainingFeatures)==0:
        return majorityVote(data,root)
    #find the index of the best feature to split on
    bestFeature = max(remainingFeatures,key = lambda index:gain(data,index))

    #信息增益为零没必要再划分
    if gain(data,bestFeature) == 0:
        return majorityVote(data,root)


    #否则建树
    root.splitFeature = bestFeature

    for dataSubset in splitData(data, bestFeature):
        aChild = Tree(parent=root)
        aChild.splitFeatureValue = dataSubset[0][0][bestFeature]
        root.children.append(aChild)
        buildDecisionTree(dataSubset, aChild, remainingFeatures - set([bestFeature]))

    return root

def decisionTree(data):
    return buildDecisionTree(data,Tree(),set(range(len(data[0][0]))))

def classify(tree,point):
    '''classify a data point by traversing the given decision tree'''
    if tree.children ==[]:
        return tree.label
    else:
        matchingChildren = [child for child in tree.children
                            if child.splitFeatureValue == point[tree.splitFeature]]
        return classify(matchingChildren[0],point)
def testClassification(data,tree):
    actualLabels = [label for point,label in data]
    predicteLabels = [classify(tree,point) for point, label in data]

    correctLabels = [(1 if a==b else 0) for a,b in zip(actualLabels,predicteLabels)];
    return float(sum[correctLabels]/len(actualLabels))


