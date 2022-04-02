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
        
    def changeLabel(self, tar, old_list, new_list):
        length = old_list.__len__()
        for i in range(length):
            new_list.append(tar if (old_list[i] == tar) else 3)

    def SVM(self, slack_C):
        acc_te = []
        acc_tr = []

        w = []
        b = []
        SV_i = []

        for i in range(3):
            # i vs rest
            tmp_Y_train = []
            tmp_Y_test = []
            self.changeLabel(i, self.Y_train, tmp_Y_train)
            self.changeLabel(i, self.Y_test, tmp_Y_test)

            mySVC = SVC(C=slack_C,decision_function_shape="ovo",kernel="linear")
            mySVC.fit(self.X_train, tmp_Y_train)

            # See results after fitting?
            w.append(mySVC.coef_)
            b.append(mySVC.intercept_)
            SV_i.append(mySVC.support_)

            # how to get train-loss and test-loss?
            acc_tr.append(mySVC.score(self.X_train, tmp_Y_train))
            # print("train: ", acc_tr[i])

            acc_te.append(mySVC.score(self.X_test, tmp_Y_test))
            # print("test: ", acc_te[i])

        train_loss = 1
        test_loss = 1
        for i in range(3): 
            train_loss -= acc_tr[i]/3.0 
            test_loss -= acc_te[i]/3.0 
            print("tr ", acc_tr[i])
            print("te ", acc_tr[i])
        # print("tr", train_loss)
        # print("te", test_loss)

        # open/create txt and write and close

        # te_x = np.array([[4.9, 3.0, 1.4, 0.2]])
        # print(w*np.asmatrix(te_x).T)
        # print(mySVC.predict(te_x))
        
        return w, b, SV_i, train_loss, test_loss
    
    def SVM_slack(self):
        train_loss = []
        test_loss = []
        acc_te = []
        acc_tr = []

        w = []
        b = []
        SV_i = []
        for i in range(3):
            # i vs rest
            tmp_Y_train = []
            tmp_Y_test = []
            self.changeLabel(i, self.Y_train, tmp_Y_train)
            self.changeLabel(i, self.Y_test, tmp_Y_test)

            for j in range(1,11):
                pass
                mySVC = SVC(C=0.1*j,decision_function_shape="ovo",kernel="linear") #decision_function_shape="ovo",  

                mySVC.fit(self.X_train, tmp_Y_train)

                # See results after fitting?
                w.append(mySVC.coef_)
                b.append(mySVC.intercept_)
                SV_i.append(mySVC.support_)

                # how to get train-loss and test-loss?
                acc_tr.append(mySVC.score(self.X_train, tmp_Y_train))
                # print("train: ", acc_tr[i])

                acc_te.append(mySVC.score(self.X_test, tmp_Y_test))
                # print("test: ", acc_te[i])

            train_loss -= acc_tr[i]/3.0 
            test_loss -= acc_te[i]/3.0 
            print("tr", train_loss)
            print("te", test_loss)   
        
        return w, b, SV_i, train_loss, test_loss
    
    def SVM_kernel_poly2(self, slack_C):
        train_loss = 0
        test_loss = 0

        w = []
        b = []
        SV_i = []    
        #########################
        ## WRITE YOUR CODE HERE##
        #########################    
        
        return w, b, SV_i, train_loss, test_loss
    
    def SVM_kernel_poly3(self, slack_C):
        train_loss = 0
        test_loss = 0

        w = []
        b = []
        SV_i = []    
        #########################
        ## WRITE YOUR CODE HERE##
        #########################    
        
        return w, b, SV_i, train_loss, test_loss
    
    def SVM_kernel_rbf(self, slack_C):
        train_loss = 0
        test_loss = 0

        w = []
        b = []
        SV_i = []    
        #########################
        ## WRITE YOUR CODE HERE##
        #########################    
        
        return w, b, SV_i, train_loss, test_loss
    
    def SVM_kernel_sigmoid(self, slack_C):
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

    Slack_C = [1e5, (i for i in range(5))]
    functions = [mySvm.SVM, mySvm.SVM_slack, mySvm.SVM_kernel_poly2, mySvm.SVM_kernel_poly3, mySvm.SVM_kernel_rbf, mySvm.SVM_kernel_sigmoid]
    # for j in range(1):
    #     list_W, list_b, list_SV_index, tr_loss, ts_loss = functions[j](Slack_C[j])


    # mySvm.SVM(1e5)
    mySvm.SVM_slack()
        

    # f = open('coding/train.txt', 'r')
    # print(svm.Y_test)
    # print(mySvm.)


    #####################################
    ## Call different SVM with value C ##
    #####################################