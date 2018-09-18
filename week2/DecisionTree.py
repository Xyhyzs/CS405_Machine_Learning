
import numpy as np
from sklearn.cross_validation import train_test_split
class Tree:
    def __init__(self, label, attr, rotest=None):
        self.root = None
        boundary = len(label) // 3
        if rotest is None:
            self.root = self.createTree(label[boundary:], attr[boundary:],
                                        np.array(range(len(attr.transpose()))), False)
            return

    @staticmethod
    def getEntroy(label, attribute):
        result = 0.0
        for this_attr in np.unique(attribute):
            label_temp, entropy = label[np.where(attribute == this_attr)[0]], 0.0
            for this_label in np.unique(label_temp):
                p = len(np.where(label_temp == this_label)[0]) / len(label_temp)
                entropy -= p * np.log2(p)
            result += len(label_temp) / len(label) * entropy
        return result

    def createTree(self, label, attribute, attr_idx, prePr, check_attr=None, check_label=None):
        node, right_count = {}, None
        max_type = np.argmax(np.bincount(label))
        if len(np.unique(label)) == 1:
            # 样本同一类
            node['type'] = label[0]
            return node
        if attribute is None or len(np.unique(attribute, axis=0)) == 1:
            #根节点调整
            node['type'] = max_type
            return node
        attr_trans = np.transpose(attribute)
        min_entropy, best_attr = np.inf, None

        if prePr:
            right_count = len(np.where(check_label == max_type)[0])
        for this_attr in attr_trans:
            entropy = self.getEntroy(label, this_attr)
            if entropy < min_entropy:
                min_entropy = entropy
                best_attr = this_attr

        branch_attr_idx = np.where((attr_trans == best_attr).all(1))[0][0]
        if prePr:
            sub_right_count = 0
            check_attr_trans = check_attr.transpose()

            for temp in np.unique(best_attr):
                branch_data_idx = np.where(best_attr == temp)[0]
                predict_label = np.argmax(np.bincount(label[branch_data_idx]))
                check_data_idx = np.where(check_attr_trans[branch_attr_idx] == temp)[0]
                check_branch_label = check_label[check_data_idx]
                sub_right_count += len(np.where(check_branch_label == predict_label)[0])
            if sub_right_count <= right_count:
                node['type'] = max_type
                return node
        values = []
        for temp in np.unique(best_attr):
            values.append(temp)
            branch_data_idx = np.where(best_attr == temp)[0]
            if len(branch_data_idx) == 0:
                new_node = {'type': np.argmax(np.bincount(label))}
            else:
                branch_label = label[branch_data_idx]
                branch_attr = np.delete(attr_trans, branch_attr_idx, axis=0).transpose()[branch_data_idx]
                new_node = self.createTree(branch_label, branch_attr,
                                           np.delete(attr_idx, branch_attr_idx, axis=0),
                                           prePr, check_attr, check_label)
            node[str(temp)] = new_node
        node['attr'] = attr_idx[branch_attr_idx]
        node['type'] = max_type
        node['values'] = values
        return node


   # 预测结果
    def predict(self, data):
        node = self.root
        while node.get('attr') is not None:
            attr = node['attr']
            node = node.get(str(data[attr]))
            if node is None:
                return None
        return node.get('type')





all_data = np.loadtxt("train.txt")# 读取数据
label = all_data[:, -1]
label = label.astype(np.int)  # label 转整数

data = np.delete(all_data, -1, axis=1)
print(data.shape,label.shape)
#划分数据集 test = 0.2
X_train,X_test,y_train,y_test = train_test_split(data,label,test_size = 0.2,random_state = 0)


print(data.shape,X_train.shape)



tree = Tree(y_train, X_train, None)


test_count = len(y_test)
test_data, test_label = X_test, y_test


times =  0 #统计

for idx in range(test_count):
    if tree.predict(test_data[idx]) == test_label[idx]:
        times += 1

print('testsize 为0.2时 正确率为 %.2f %%' % (times * 100 / test_count))

#划分数据集 test = 0.4
X_train,X_test,y_train,y_test = train_test_split(data,label,test_size = 0.4,random_state = 0)


# print(data.shape,X_train.shape)



tree = Tree(y_train, X_train, None)


test_count = len(y_test)
test_data, test_label = X_test, y_test


times =  0 #统计

for idx in range(test_count):
    if tree.predict(test_data[idx]) == test_label[idx]:
        times += 1

print('testsize 为0.4时 正确率为 %.2f %%' % (times * 100 / test_count))

#划分数据集 test = 0.6
X_train,X_test,y_train,y_test = train_test_split(data,label,test_size = 0.6,random_state = 0)


# print(data.shape,X_train.shape)



tree = Tree(y_train, X_train, None)


test_count = len(y_test)
test_data, test_label = X_test, y_test


times =  0 #统计

for idx in range(test_count):
    if tree.predict(test_data[idx]) == test_label[idx]:
        times += 1

print('testsize 为0.6时 正确率为 %.2f %%' % (times * 100 / test_count))

#划分数据集 test = 0.8
X_train,X_test,y_train,y_test = train_test_split(data,label,test_size = 0.8,random_state = 0)


# print(data.shape,X_train.shape)



tree = Tree(y_train, X_train, None)


test_count = len(y_test)
test_data, test_label = X_test, y_test


times =  0 #统计

for idx in range(test_count):
    if tree.predict(test_data[idx]) == test_label[idx]:
        times += 1

print('testsize 为0.8时 正确率为 %.2f %%' % (times * 100 / test_count))

