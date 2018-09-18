import numpy as np



#计算信息增益

def get_info_entropy(label,attr):#label 标签 attr:用于数据划分的属性集合
    result  = 0.0
    for temp_attr in np.unique(attr):
        sublabel,entropy = label[np.where(attr == temp_attr)[0]],0.0
        for temp_label in np.unique(sublabel):
             p  = len(np.where(sublabel == temp_label)[0])/len(sublabel)
             entropy -=p*np.log2(p)
        result += len(sublabel)/len(label)*entropy
    return result

def create_tree(self, label ,attr, attr_idx,pre_pruning, check_attr=None, check_label = None):
    node,right_count = {},None
    max_type = np.argmax(np.bincount(label))
    if len(np.unipue(label