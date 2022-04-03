import re
import numpy as np
from sklearn import datasets, model_selection, metrics
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import svm, datasets

def maxProb(a, b, c):
    max = a
    if (b > a): max = b
    if (c > max): max = c

    if (max == a): return 0
    if (max == b): return 1
    if (max == c): return 2

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

    def change_label(self, tar, old_list, new_list):
        length = old_list.__len__()
        for i in range(length):
            new_list.append(1 if (old_list[i] == tar) else -1)

    def train_model(self, SVCs, w, b, SV_i, model="kernel"):
        for i in range(3):
            tmp_Y_train = []
            tmp_Y_test = []
            self.change_label(i, self.Y_train, tmp_Y_train)
            self.change_label(i, self.Y_test, tmp_Y_test)
            SVCs[i].fit(self.X_train, tmp_Y_train)

            if model == "no_kernel":
                w.append(SVCs[i].coef_)
            b.append(SVCs[i].intercept_)
            SV_i.append(SVCs[i].support_)  
        return 

    def get_err(self, SVCs):
        err_tr = 0
        err_te = 0
        pred_tr = []
        pred_te = []     

        # for training data
        for j in range(120):
            x_j = np.array(self.X_train[j]).reshape(1, -1)
            d0 = SVCs[0].decision_function(x_j); d0 = d0[0]
            d1 = SVCs[1].decision_function(x_j); d1 = d1[0]
            d2 = SVCs[2].decision_function(x_j); d2 = d2[0]
            pred_tr.append(maxProb(d0, d1, d2))

        for j_ in range(120):
            if (pred_tr[j_] != self.Y_train[j_]):
                err_tr += 1
        err_tr = err_tr/120.0

        # for testing data
        for k in range(30):
            x_k = np.array([self.X_test[k]]).reshape(1, -1)
            d0 = SVCs[0].decision_function(x_k); d0 = d0[0]
            d1 = SVCs[1].decision_function(x_k); d1 = d1[0]
            d2 = SVCs[2].decision_function(x_k); d2 = d2[0]
            pred_te.append(maxProb(d0, d1, d2))

        for k_ in range(30):
            if (pred_te[k_] != self.Y_test[k_]):
                err_te += 1
        err_te = err_te/30.0

        # print(pred_tr)
        # print(pred_te)
        
        # te_x = np.array([[4.9, 3.0, 1.4, 0.2]])
        # print(w*np.asmatrix(te_x).T)
        # print(mySVC.predict(te_x))

        return err_tr, err_te

    def write_file(self, fiename, train_error, test_error, w=[], b=[], SV=[], model="no_kernel"):
        my_file = open(fiename,'w')
        my_file.write(str(train_error)+"\n")
        my_file.write(str(test_error)+"\n")
        for i in range(3):
            if (model=="no_kernel"): 
                w_str = str(w[i][0][0]) + ", " + str(w[i][0][1]) + ", " + str(w[i][0][2])
                my_file.write(w_str+"\n")
            my_file.write(str(b[i][0])+"\n")

            tmp = SV[i]
            l = len(tmp)
            SV_str = ""
            for j in range(l-1):
                SV_str = SV_str + str(tmp[j]) + ", "
            SV_str += str(tmp[l-1])
            my_file.write(SV_str+"\n")
        return

    def SVM(self):
        mySVC0 = SVC(C=1e5,decision_function_shape="ovo",kernel="linear")
        mySVC1 = SVC(C=1e5,decision_function_shape="ovo",kernel="linear")
        mySVC2 = SVC(C=1e5,decision_function_shape="ovo",kernel="linear")
        mySVCs = [mySVC0, mySVC1, mySVC2]

        # fit and store the model
        w = []; b = []; SV_i = []
        self.train_model(mySVCs, w, b, SV_i, "no_kernel")

        # compute the train-loss and test-loss
        err_tr, err_te = self.get_err(mySVCs)
        
        # generate the output file
        filename = "SVM_linear.txt"
        self.write_file(filename, err_tr, err_te, w, b, SV_i, "no_kernel")

        # print(err_tr)
        # print(err_te)
        # for i in range(3):
        #     for j in range(3):
        #         print(w[i][0][j], end = ", ")
        #     print(w[i][0][3])
        #     print(b[i][0])
        #     tmp = SV_i[i]
        #     l = len(tmp)
        #     for j in range(l-1):
        #         print(tmp[j], end = ", ")
        #     print(tmp[l-1])

        return err_tr, err_te, SV_i
    
    def SVM_slack(self, C):
        mySVC0 = SVC(C=C,decision_function_shape="ovo",kernel="linear")
        mySVC1 = SVC(C=C,decision_function_shape="ovo",kernel="linear")
        mySVC2 = SVC(C=C,decision_function_shape="ovo",kernel="linear")
        mySVCs = [mySVC0, mySVC1, mySVC2]

        # See results after fitting?
        w = []; b = []; SV_i = []
        self.train_model(mySVCs, w, b, SV_i, "no_kernel")            

        # how to get train-loss and test-loss?
        err_tr, err_te = self.get_err(mySVCs)

        # generate the output file
        filename = "SVM_slack_"+ str(C) +".txt"
        self.write_file(filename, err_tr, err_te, w, b, SV_i, "no_kernel")
        return err_tr, err_te, SV_i
    
    def SVM_kernel_poly2(self, C=1):
        mySVC0 = SVC(C=1,decision_function_shape="ovo",kernel="poly",degree=2,gamma=1,coef0=0) # sigma?
        mySVC1 = SVC(C=1,decision_function_shape="ovo",kernel="poly",degree=2,gamma=1,coef0=0)
        mySVC2 = SVC(C=1,decision_function_shape="ovo",kernel="poly",degree=2,gamma=1,coef0=0)
        mySVCs = [mySVC0, mySVC1, mySVC2]

        # See results after fitting?
        w = []; b = []; SV_i = []
        self.train_model(mySVCs, w, b, SV_i)

        # # how to get train-loss and test-loss?
        err_tr, err_te = self.get_err(mySVCs)

        # generate the output file
        filename = "SVM_poly2.txt"
        self.write_file(filename, err_tr, err_te, w, b, SV_i, "kernel")

        return err_tr, err_te, SV_i
    
    def SVM_kernel_poly3(self, C=1):
        mySVC0 = SVC(C=1,decision_function_shape="ovo",kernel="poly",degree=3,gamma=1,coef0=0) # sigma?
        mySVC1 = SVC(C=1,decision_function_shape="ovo",kernel="poly",degree=3,gamma=1,coef0=0)
        mySVC2 = SVC(C=1,decision_function_shape="ovo",kernel="poly",degree=3,gamma=1,coef0=0)
        mySVCs = [mySVC0, mySVC1, mySVC2]

        # See results after fitting?
        w = []; b = []; SV_i = []
        self.train_model(mySVCs, w, b, SV_i)

        # how to get train-loss and test-loss?
        err_tr, err_te = self.get_err(mySVCs)

        # generate the output file
        filename = "SVM_poly3.txt"
        self.write_file(filename, err_tr, err_te, w, b, SV_i, "kernel")

        return err_tr, err_te, SV_i
    
    def SVM_kernel_rbf(self, C=1):
        mySVC0 = SVC(decision_function_shape="ovo",kernel="rbf",gamma=0.5) # sigma^2 = 1/gamma 
        mySVC1 = SVC(decision_function_shape="ovo",kernel="rbf",gamma=0.5)
        mySVC2 = SVC(decision_function_shape="ovo",kernel="rbf",gamma=0.5)
        mySVCs = [mySVC0, mySVC1, mySVC2]

        # See results after fitting?
        w = []; b = []; SV_i = []
        self.train_model(mySVCs, w, b, SV_i)

        # how to get train-loss and test-loss?
        err_tr, err_te = self.get_err(mySVCs)

        # generate the output file
        filename = "SVM rbf.txt"
        self.write_file(filename, err_tr, err_te, w, b, SV_i, "kernel")

        return err_tr, err_te, SV_i
    
    def SVM_kernel_sigmoid(self, C=1):
        mySVC0 = SVC(decision_function_shape="ovo",kernel="sigmoid",gamma="auto") # sigma^2 = 1/gamma 
        mySVC1 = SVC(decision_function_shape="ovo",kernel="sigmoid",gamma="auto")
        mySVC2 = SVC(decision_function_shape="ovo",kernel="sigmoid",gamma="auto")
        mySVCs = [mySVC0, mySVC1, mySVC2]

        # See results after fitting?
        w = []; b = []; SV_i = []
        self.train_model(mySVCs, w, b, SV_i)

        # how to get train-loss and test-loss?
        err_tr, err_te = self.get_err(mySVCs)   
        
        # generate the output file
        filename = "SVM sigmoid.txt"
        self.write_file(filename, err_tr, err_te, w, b, SV_i, "kernel")

        return err_tr, err_te, SV_i


if __name__ =='__main__':
            
    mySvm = SVM("coding/train.txt", "coding/test.txt")#!!!!!!!!!!!!!!!!!!!!
    mySvm.read_data()

    mySvm.SVM()
    for i in range(10):
        mySvm.SVM_slack((i+1)*0.1)
    mySvm.SVM_kernel_poly2()
    mySvm.SVM_kernel_poly3()
    mySvm.SVM_kernel_rbf()
    mySvm.SVM_kernel_sigmoid()
        
