import re
import numpy as np
from sklearn import datasets, model_selection, metrics
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import svm, datasets

class SVM(object):

    def __init__(self, training_dataset_, test_dataset_):
        self.training_dataset = training_dataset_
        self.test_dataset = test_dataset_
        self.classes = {}
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        self.support_indecies = None
        self.train_errors = None
        self.val_errors = None
        self.train_loss = None
        self.val_loss = None
        
    def read_data(self):
        # 类似的事情做两次
        # rows 将每一行split后根据空格分开
        # names是3*class name， Y_train是一堆0 1 2
        # # self.X_train nxd 样本数量x维度
        f = open(self.training_dataset, 'r')
        rows = list(re.split(' ', row) for row in re.split('\n', f.read())[:-1]) 
        names, self.Y_train = np.unique(list(row[-1] for row in rows), return_inverse=True) 
        self.X_train = np.empty((0,4), float) 
        f.close()
        for row in rows:
            self.X_train = np.append(self.X_train, np.array([np.array(row[:-1]).astype(float)]), axis = 0) 
        
        f = open(self.test_dataset, 'r')
        rows = list(re.split(' ', row) for row in re.split('\n', f.read())[:-1])
        names, self.Y_test = np.unique(list(row[-1] for row in rows), return_inverse=True)
        self.X_test = np.empty((0,4), float)
        f.close()
        for row in rows:
            self.X_test = np.append(self.X_test, np.array([np.array(row[:-1]).astype(float)]), axis = 0)
        
    def mergeData(self, tar, new_list):
        length = self.Y_train.__len__()
        for i in range(length):
            new_list.append(tar if (self.Y_train[i] == tar) else 3)

    def SVM(self):
        train_loss = 0
        test_loss = 0
        support_vectors = 0

        w = []
        b = []
        SV_i = []
        for i in range(3):
            # i vs rest
            tmp_Y_train = []
            self.mergeData(i, tmp_Y_train)

            mySVC = SVC(C=1e5, decision_function_shape="ovo", kernel="linear")
            mySVC.fit(self.X_train, tmp_Y_train)

            # See results after fitting?
            w.append(np.asmatrix(mySVC.coef_))
            b.append(mySVC.intercept_)
            SV_i.append(mySVC.support_)


        # te_x = np.array([[4.9, 3.0, 1.4, 0.2]])
        # print(w*np.asmatrix(te_x).T)
        # print(mySVC.predict(te_x))
        
        return w, b, SV_i, train_loss, test_loss, support_vectors
    
    def SVM_slack(C):
        train_loss = 0
        test_loss = 0
        support_vectors = 0

        w = []
        b = []
        SV_i = []    
        #########################
        ## WRITE YOUR CODE HERE##
        #########################    
        
        return w, b, SV_i, train_loss, test_loss, support_vectors
    
    def SVM_kernel_poly2(C):
        train_loss = 0
        test_loss = 0
        support_vectors = 0

        w = []
        b = []
        SV_i = []    
        #########################
        ## WRITE YOUR CODE HERE##
        #########################    
        
        return w, b, SV_i, train_loss, test_loss, support_vectors
    
    def SVM_kernel_poly3(C):
        train_loss = 0
        test_loss = 0
        support_vectors = 0

        w = []
        b = []
        SV_i = []    
        #########################
        ## WRITE YOUR CODE HERE##
        #########################    
        
        return w, b, SV_i, train_loss, test_loss, support_vectors
    
    def SVM_kernel_rbf(C):
        train_loss = 0
        test_loss = 0
        support_vectors = 0

        w = []
        b = []
        SV_i = []    
        #########################
        ## WRITE YOUR CODE HERE##
        #########################    
        
        return w, b, SV_i, train_loss, test_loss, support_vectors
    
    def SVM_kernel_sigmoid(C):
        train_loss = 0
        test_loss = 0
        support_vectors = 0

        w = []
        b = []
        SV_i = []    
        #########################
        ## WRITE YOUR CODE HERE##
        #########################    
        
        return w, b, SV_i, train_loss, test_loss, support_vectors
    
mySvm = SVM("coding/train.txt", "coding/test.txt")#!!!!!!!!!!!!!!!!!!!!
mySvm.read_data()
mySvm.SVM()
# f = open('coding/train.txt', 'r')
# print(svm.Y_test)
# print(mySvm.)


#####################################
## Call different SVM with value C ##
#####################################