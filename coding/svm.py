import re
import numpy as np
from sklearn import datasets, model_selection, metrics
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import svm, datasets

def maxProb(a, b, c):
    max = 0
    if (b > max):
        max = 1
    if (c > max):
        max = 2
    return max

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

    def changeLabel(self, tar, old_list, new_list):
        length = old_list.__len__()
        for i in range(length):
            new_list.append(1 if (old_list[i] == tar) else -1)

    def get_err(self, SVCs):
        err_tr = 0
        err_te = 0
        for i in range(120):
            pass
        for j in range(30):
            pass
        SVCs[0]
        SVCs[1]
        SVCs[2]
        return err_tr, err_te

    def SVM(self):
        mySVC0 = SVC(C=1e5,decision_function_shape="ovo",kernel="linear")
        mySVC1 = SVC(C=1e5,decision_function_shape="ovo",kernel="linear")
        mySVC2 = SVC(C=1e5,decision_function_shape="ovo",kernel="linear")
        mySVCs = [mySVC0, mySVC1, mySVC2]

        # i vs rest
        # See results after fitting?
        w = []
        b = []
        SV_i = []

        for i in range(3):
            tmp_Y_train = []
            tmp_Y_test = []
            self.changeLabel(i, self.Y_train, tmp_Y_train)
            self.changeLabel(i, self.Y_test, tmp_Y_test)
            mySVCs[i].fit(self.X_train, tmp_Y_train)

            w.append(mySVCs[i].coef_)
            b.append(mySVCs[i].intercept_)
            SV_i.append(mySVCs[i].support_)            

        # how to get train-loss and test-loss?
        pred_tr = []
        pred_te = []
        err_tr = 0
        err_te = 0

        # for training data
        for j in range(120):
            x_j = np.array(self.X_train[j]).reshape(1, -1)
            d0 = mySVCs[0].decision_function(x_j); d0 = d0[0]
            d1 = mySVCs[1].decision_function(x_j); d1 = d1[0]
            d2 = mySVCs[2].decision_function(x_j); d2 = d2[0]
            pred_tr.append(maxProb(d0, d1, d2))
        
        for j_ in range(120):
            if (pred_tr[j_] != self.Y_train[j_]):
                err_tr += 1
        err_tr = err_tr/120.0

        # for training data
        for k in range(30):
            x_k = np.array([self.X_test[k]]).reshape(1, -1)
            d0 = mySVCs[0].decision_function(x_k); d0 = d0[0]
            d1 = mySVCs[1].decision_function(x_k); d1 = d1[0]
            d2 = mySVCs[2].decision_function(x_k); d2 = d2[0]
            pred_te.append(maxProb(d0, d1, d2))
        
        for k_ in range(30):
            if (pred_te[k_] != self.Y_test[k_]):
                err_te += 1
        err_te = err_te/30.0

        # te_x = np.array([[4.9, 3.0, 1.4, 0.2]])
        # print(w*np.asmatrix(te_x).T)
        # print(mySVC.predict(te_x))
        
        # print(err_tr)
        # print(err_te)
        # for i in range(3):
        #     print(w[i], "\n", b[i])
        return w, b, SV_i, err_tr, err_te
    
    def SVM_slack(self, slack_C):
        mySVC0 = SVC(slack_C,decision_function_shape="ovo",kernel="linear")
        mySVC1 = SVC(slack_C,decision_function_shape="ovo",kernel="linear")
        mySVC2 = SVC(slack_C,decision_function_shape="ovo",kernel="linear")
        mySVCs = [mySVC0, mySVC1, mySVC2]

        # i vs rest
        # See results after fitting?
        w = []
        b = []
        SV_i = []
        for i in range(3):
            tmp_Y_train = []
            tmp_Y_test = []
            self.changeLabel(i, self.Y_train, tmp_Y_train)
            self.changeLabel(i, self.Y_test, tmp_Y_test)
            mySVCs[i].fit(self.X_train, tmp_Y_train)

            w.append(mySVCs[i].coef_)
            b.append(mySVCs[i].intercept_)
            SV_i.append(mySVCs[i].support_)            

        # how to get train-loss and test-loss?
        pred_tr = []
        pred_te = []
        err_tr = 0
        err_te = 0

        # for training data
        for j in range(10):
            x_j = np.array([self.X_train[j]])
            d0 = mySVCs[0].decision_function(x_j); d0 = d0[0]
            d1 = mySVCs[1].decision_function(x_j); d1 = d1[0]
            d2 = mySVCs[2].decision_function(x_j); d2 = d2[0]
            pred_tr.append(maxProb(d0, d1, d2))
            # print(pre_j)
        
        for j_ in range(120):
            if (pred_tr[j_] != self.Y_train[j_]):
                err_tr += 1
        err_tr = err_tr/120.0

        # for training data
        for k in range(30):
            x_k = np.array([self.Y_test[k]])
            d0 = mySVCs[0].decision_function(x_k); d0 = d0[0]
            d1 = mySVCs[1].decision_function(x_k); d1 = d1[0]
            d2 = mySVCs[2].decision_function(x_k); d2 = d2[0]
            pred_te.append(maxProb(d0, d1, d2))
            #print((d0, d1, d2))
        
        for k_ in range(30):
            if (pred_te[k_] != self.Y_test[k_]):
                err_te += 1
        err_te = err_te/30.0

        # te_x = np.array([[4.9, 3.0, 1.4, 0.2]])
        # print(w*np.asmatrix(te_x).T)
        # print(mySVC.predict(te_x))
        
        return w, b, SV_i, err_tr, err_te
    
    def SVM_kernel_poly2(self):
        mySVC0 = SVC(C=1,decision_function_shape="ovo",kernel="poly",degree=2,gamma="auto") # sigma?
        mySVC1 = SVC(C=1,decision_function_shape="ovo",kernel="poly",degree=2,gamma="auto")
        mySVC2 = SVC(C=1,decision_function_shape="ovo",kernel="poly",degree=2,gamma="auto")
        mySVCs = [mySVC0, mySVC1, mySVC2]

        # i vs rest
        # See results after fitting?
        b = []
        SV_i = []
        for i in range(3):
            tmp_Y_train = []
            tmp_Y_test = []
            self.changeLabel(i, self.Y_train, tmp_Y_train)
            self.changeLabel(i, self.Y_test, tmp_Y_test)
            mySVCs[i].fit(self.X_train, tmp_Y_train)

            b.append(mySVCs[i].intercept_)
            SV_i.append(mySVCs[i].support_)

        # # how to get train-loss and test-loss?
        pred_tr = []
        pred_te = []
        err_tr = 0
        err_te = 0

        # for training data
        for j in range(120):
            x_j = np.array([self.X_train[j]]).reshape(1,-1)
            d0 = mySVCs[0].decision_function(x_j); d0 = d0[0]
            d1 = mySVCs[1].decision_function(x_j); d1 = d1[0]
            d2 = mySVCs[2].decision_function(x_j); d2 = d2[0]
            pred_tr.append(maxProb(d0, d1, d2))
            # print(pre_j)
        
        for j_ in range(120):
            if (pred_tr[j_] != self.Y_train[j_]):
                err_tr += 1
        err_tr = err_tr/120.0

        # for training data
        for k in range(30):
            x_k = np.array([self.X_test[k]]).reshape(1,-1)
            d0 = mySVCs[0].decision_function(x_k); d0 = d0[0]
            d1 = mySVCs[1].decision_function(x_k); d1 = d1[0]
            d2 = mySVCs[2].decision_function(x_k); d2 = d2[0]
            pred_te.append(maxProb(d0, d1, d2))
            #print((d0, d1, d2))
        
        for k_ in range(30):
            if (pred_te[k_] != self.Y_test[k_]):
                err_te += 1
        err_te = err_te/30.0

        print(err_tr)
        print(err_te)
        print(SV_i[0])
        print(SV_i[1])
        print(SV_i[2])
        print(b)

        # te_x = np.array([[4.9, 3.0, 1.4, 0.2]])
        # print(w*np.asmatrix(te_x).T)
        # print(mySVC.predict(te_x))
        
        return b, SV_i, err_tr, err_te
    
    def SVM_kernel_poly3(self):
        mySVC0 = SVC(C=1,decision_function_shape="ovo",kernel="poly",degree=3,gamma="auto") # sigma?
        mySVC1 = SVC(C=1,decision_function_shape="ovo",kernel="poly",degree=3,gamma="auto")
        mySVC2 = SVC(C=1,decision_function_shape="ovo",kernel="poly",degree=3,gamma="auto")
        mySVCs = [mySVC0, mySVC1, mySVC2]

        # i vs rest
        # See results after fitting?
        b = []
        SV_i = []
        for i in range(3):
            tmp_Y_train = []
            tmp_Y_test = []
            self.changeLabel(i, self.Y_train, tmp_Y_train)
            self.changeLabel(i, self.Y_test, tmp_Y_test)
            mySVCs[i].fit(self.X_train, tmp_Y_train)

            b.append(mySVCs[i].intercept_)
            SV_i.append(mySVCs[i].support_)

        # # how to get train-loss and test-loss?
        pred_tr = []
        pred_te = []
        err_tr = 0
        err_te = 0

        # for training data
        for j in range(120):
            x_j = np.array([self.X_train[j]]).reshape(1,-1)
            d0 = mySVCs[0].decision_function(x_j); d0 = d0[0]
            d1 = mySVCs[1].decision_function(x_j); d1 = d1[0]
            d2 = mySVCs[2].decision_function(x_j); d2 = d2[0]
            pred_tr.append(maxProb(d0, d1, d2))
            # print(pre_j)
        
        for j_ in range(120):
            if (pred_tr[j_] != self.Y_train[j_]):
                err_tr += 1
        err_tr = err_tr/120.0

        # for training data
        for k in range(30):
            x_k = np.array([self.X_test[k]]).reshape(1,-1)
            d0 = mySVCs[0].decision_function(x_k); d0 = d0[0]
            d1 = mySVCs[1].decision_function(x_k); d1 = d1[0]
            d2 = mySVCs[2].decision_function(x_k); d2 = d2[0]
            pred_te.append(maxProb(d0, d1, d2))
            #print((d0, d1, d2))
        
        for k_ in range(30):
            if (pred_te[k_] != self.Y_test[k_]):
                err_te += 1
        err_te = err_te/30.0

        print(err_tr)
        print(err_te)
        print(b)
        print(SV_i[0])
        print(SV_i[1])
        print(SV_i[2])

        # te_x = np.array([[4.9, 3.0, 1.4, 0.2]])
        # print(w*np.asmatrix(te_x).T)
        # print(mySVC.predict(te_x))
        return b, SV_i, err_tr, err_te
    
    def SVM_kernel_rbf(self):
        mySVC0 = SVC(decision_function_shape="ovo",kernel="rbf",gamma=0.5) # sigma^2 = 1/gamma 
        mySVC1 = SVC(decision_function_shape="ovo",kernel="rbf",gamma=0.5)
        mySVC2 = SVC(decision_function_shape="ovo",kernel="rbf",gamma=0.5)
        mySVCs = [mySVC0, mySVC1, mySVC2]

        # i vs rest
        # See results after fitting?
        b = []
        SV_i = []
        for i in range(3):
            tmp_Y_train = []
            tmp_Y_test = []
            self.changeLabel(i, self.Y_train, tmp_Y_train)
            self.changeLabel(i, self.Y_test, tmp_Y_test)
            mySVCs[i].fit(self.X_train, tmp_Y_train)

            b.append(mySVCs[i].intercept_)
            SV_i.append(mySVCs[i].support_)

        # # how to get train-loss and test-loss?
        pred_tr = []
        pred_te = []
        err_tr = 0
        err_te = 0

        # for training data
        for j in range(120):
            x_j = np.array([self.X_train[j]]).reshape(1,-1)
            d0 = mySVCs[0].decision_function(x_j); d0 = d0[0]
            d1 = mySVCs[1].decision_function(x_j); d1 = d1[0]
            d2 = mySVCs[2].decision_function(x_j); d2 = d2[0]
            pred_tr.append(maxProb(d0, d1, d2))
            # print(pre_j)
        
        for j_ in range(120):
            if (pred_tr[j_] != self.Y_train[j_]):
                err_tr += 1
        err_tr = err_tr/120.0

        # for training data
        for k in range(30):
            x_k = np.array([self.X_test[k]]).reshape(1,-1)
            d0 = mySVCs[0].decision_function(x_k); d0 = d0[0]
            d1 = mySVCs[1].decision_function(x_k); d1 = d1[0]
            d2 = mySVCs[2].decision_function(x_k); d2 = d2[0]
            pred_te.append(maxProb(d0, d1, d2))
            #print((d0, d1, d2))
        
        for k_ in range(30):
            if (pred_te[k_] != self.Y_test[k_]):
                err_te += 1
        err_te = err_te/30.0

        print(err_tr)
        print(err_te)
        print(b)
        print(SV_i[0])
        print(SV_i[1])
        print(SV_i[2])

        # te_x = np.array([[4.9, 3.0, 1.4, 0.2]])
        # print(w*np.asmatrix(te_x).T)
        # print(mySVC.predict(te_x))
        return b, SV_i, err_tr, err_te
    
    def SVM_kernel_sigmoid(self):
        train_loss = 0
        test_loss = 0

        w = []
        b = []
        SV_i = []    
        #########################
        ## WRITE YOUR CODE HERE##
        #########################    
        
        return w, b, SV_i, train_loss, test_loss
    
if __name__ =='__main__':
            
    mySvm = SVM("coding/train.txt", "coding/test.txt")#!!!!!!!!!!!!!!!!!!!!
    mySvm.read_data()

    # mySvm.SVM()
    for i in range(10):
        pass
        # mySvm.SVM_slack((i+1)*0.1)
    # mySvm.SVM_kernel_poly2()
    # mySvm.SVM_kernel_poly3()
    mySvm.SVM_kernel_rbf()
        
    # open/create txt and write and close
    # f = open('coding/train.txt', 'r')
    # print(svm.Y_test)
    # print(mySvm.)


    #####################################
    ## Call different SVM with value C ##
    #####################################