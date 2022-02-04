from itertools import combinations
from math import fabs
from matplotlib.pyplot import cla, figure, plot, savefig as plt
import numpy as np
import time
from numpy.core.defchararray import mod
from numpy.core.fromnumeric import size
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.utils import validation
import tensorflow as tf
import graphviz
import keras
import tensorflow_addons as tfa
from tqdm import tqdm
from tqdm import trange

from time import sleep


import seaborn as sns
from sklearn import metrics
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifierCV

from sklearn.model_selection import KFold

from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier


def ImportFeatureFile():
    print("↓↓↓ import traing data and test data ↓↓↓")
    print()
    raw_data = open("data_file/TrainingSet.csv",mode="r",encoding="utf-8",errors="ignore")
    dataset = np.loadtxt(raw_data, delimiter='\t')

    X_train = dataset[:,0:22]
    Y_train = dataset[:,-1]

    raw_data = open("data_file/TestSet.csv",mode="r",encoding="utf-8",errors="ignore")
    dataset = np.loadtxt(raw_data, delimiter='\t')

    X_test = dataset[:,0:22]
    Y_test = dataset[:,-1]
    print("Success import data in 'data_file' Folder ", "training data set size : ",size(Y_train),",test data set size : ",size(Y_test))
    print()
    return X_train, Y_train, X_test, Y_test

def Learning(FinalModel,X_test,expected):
    start = time.time()
    predicted = FinalModel.predict(X_test)
    print(FinalModel.__class__.__name__, " accuracy : {0:.4f}".format(accuracy_score(Y_test,predicted)))
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    end = time.time() - start
    return [FinalModel.__class__.__name__,end,accuracy_score(Y_test,predicted)]
    
def IndependentModel(X_train, Y_train, X_test, Y_test):
    print("↓↓↓ Independent Machine Learning Model ↓↓↓")
    print()
    data_result = pd.DataFrame(columns=('model','time','accurancy'))
    data_result.loc[0] = Learning(DecisionTreeClassifier(max_depth=100,min_samples_split=2,criterion='gini',splitter='best').fit(X_train,Y_train),X_test,Y_test)
    data_result.loc[1] = Learning(GradientBoostingClassifier(max_depth=100,min_samples_split=2,learning_rate=0.1).fit(X_train,Y_train),X_test,Y_test)
    data_result.loc[2] = Learning(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=100,min_samples_split=2,criterion='gini',splitter='best'),n_estimators=100,random_state=1).fit(X_train,Y_train),X_test,Y_test)
    data_result.loc[3] = Learning(RandomForestClassifier(max_depth=100,n_estimators=200).fit(X_train,Y_train),X_test,Y_test)
    data_result.loc[4] = Learning(ExtraTreeClassifier().fit(X_train,Y_train),X_test,Y_test)
    data_result.loc[5] = Learning(AdaBoostClassifier(base_estimator=ExtraTreeClassifier()).fit(X_train,Y_train),X_test,Y_test)
    data_result.loc[6] = Learning(BaggingClassifier(base_estimator=ExtraTreeClassifier()).fit(X_train,Y_train),X_test,Y_test)
    data_result.loc[7] = Learning(HistGradientBoostingClassifier().fit(X_train,Y_train),X_test,Y_test)
    print()
    print("Save the accuracy excel file for Independent Machine Learning Model in the 'Result' Folder")
    data_result.to_csv('Result/Independent Model Result.csv',index=False)
    print()

def ModelKFold(FinalModel,X_train, Y_train, X_test,n_fold):
    print(FinalModel.__class__.__name__)
    kfold = KFold(n_splits=n_fold)
    train_fold_predicted = np.zeros((X_train.shape[0],1))
    test_predicted = np.zeros((X_test.shape[0],n_fold))

    for folder_counter, (train_index, valid_index) in enumerate(kfold.split(X_train)):

        X_train_ = X_train[train_index]
        Y_train_ = Y_train[train_index]
        X_test_ = X_train[valid_index]

        FinalModel.fit(X_train_,Y_train_)
        train_fold_predicted[valid_index, :] = FinalModel.predict(X_test_).reshape(-1,1)
        test_predicted[:,folder_counter] = FinalModel.predict(X_test)
    
    test_predicted_mean = np.mean(test_predicted,axis=1).reshape(-1,1)

    return train_fold_predicted,test_predicted_mean

def StackingModel(X_train, Y_train, X_test, Y_test,n_fold):

    print("↓↓↓ Independent Machine Learning Model KFold ↓↓↓")
    print()
    DT_train, DT_test = ModelKFold(DecisionTreeClassifier(max_depth=100,min_samples_split=2,criterion='gini',splitter='best'),X_train, Y_train, X_test, n_fold)
    GBC_train, GBC_test = ModelKFold(GradientBoostingClassifier(max_depth=100,min_samples_split=2,learning_rate=0.1),X_train, Y_train, X_test,n_fold)
    ABC_train, ABC_test = ModelKFold(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=100,min_samples_split=2,criterion='gini',splitter='best'),n_estimators=100,random_state=1),X_train, Y_train, X_test,n_fold)
    RF_train, RF_test = ModelKFold(RandomForestClassifier(max_depth=100,n_estimators=200),X_train, Y_train, X_test, n_fold)
    ET_train,ET_test = ModelKFold(ExtraTreeClassifier(),X_train, Y_train, X_test,n_fold)
    AET_train,AET_test = ModelKFold(AdaBoostClassifier(base_estimator=ExtraTreeClassifier()),X_train,Y_train,X_test,n_fold)
    BET_train,BET_test = ModelKFold(BaggingClassifier(base_estimator=ExtraTreeClassifier()),X_train,Y_train,X_test,n_fold)
    HGB_train,HGB_test = ModelKFold(HistGradientBoostingClassifier(),X_train, Y_train, X_test,n_fold)

    DT = np.array(["DT",DT_train,DT_test],dtype=object)
    RF = np.array(["RF",RF_train,RF_test],dtype=object)
    ABC = np.array(["ABC",ABC_train,ABC_test],dtype=object)
    GBC = np.array(["GBC",GBC_train,GBC_test],dtype=object)
    HGB = np.array(["HGB",HGB_train,HGB_test],dtype=object)
    ET = np.array(["ET",ET_train,ET_test],dtype=object)
    AET = np.array(["AET",AET_train,AET_test],dtype=object)
    BET = np.array(["BET",BET_train,BET_test],dtype=object)

    print()

    combinationModel = np.array([DT,RF,ABC,GBC,HGB,ET,AET,BET])
    model1 = np.array(["DecisionTreeClassifier","RandomForestClassifier","AdaBoostClassifier","GradientBoostingClassifier","HistGradientBoostingClassifier","ExtraTreeClassifier","Ada-ExtraTreeClassifier","Bagging-ExtraTreeClassifier"])
    FinalModel = np.array([DecisionTreeClassifier(),RandomForestClassifier(),AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),GradientBoostingClassifier(),HistGradientBoostingClassifier(),ExtraTreeClassifier(),AdaBoostClassifier(base_estimator=ExtraTreeClassifier()),BaggingClassifier(base_estimator=ExtraTreeClassifier())])

    data_result1 = pd.DataFrame(columns=('Final Model','model1','model2','model3','time','accuracy'))
    data_result2 = pd.DataFrame(columns=('Final Model','model1','model2','model3','model4','model5','time','accuracy'))
    data_result3 = pd.DataFrame(columns=('Final Model','model1','model2','model3','model4','model5','model6','model7','time','accuracy'))

    MaxAccurancy = 0
    Num = 0
    print("↓↓↓ Start 3 Combination Model ↓↓↓")
    print()
    for i in tqdm(range(0,len(FinalModel)), desc = 'Processing'):
        for j in range(0,len(combinationModel)):
            for l in range (j+1,len(combinationModel)):
                for k in range(l+1,len(combinationModel)):
                    start = time.time()
                    FinalModelTrain = np.concatenate((combinationModel[j][1],combinationModel[l][1],combinationModel[k][1]),axis=1)
                    FinalModelTest = np.concatenate((combinationModel[j][2],combinationModel[l][2],combinationModel[k][2]),axis=1)
                    Final_Model = FinalModel[i].fit(FinalModelTrain, Y_train)
                    FinalModelPredict = Final_Model.predict(FinalModelTest)
                    FinalModelAccurancy = accuracy_score(Y_test,FinalModelPredict)
                    end = time.time() - start
                    SaveResultData = [model1[i],combinationModel[j][0],combinationModel[l][0],combinationModel[k][0],end,FinalModelAccurancy]
                    data_result1.loc[Num] = SaveResultData
                    Num += 1
                    MaxAccurancy = max(MaxAccurancy,FinalModelAccurancy)
                    time.sleep(0.01)

    data_result1.to_csv('Result/3 Combination Models.csv',index=False)
    print("Save the accuracy excel file for 3 machine learning combinations in the 'Result' Folder")
    print("max accuracy score {0:.3f}".format(MaxAccurancy))

    MaxAccurancy = 0
    Num = 0
    print()
    print("↓↓↓ Start 5 Combination Model ↓↓↓")
    print()
    for i in tqdm(range(0,len(FinalModel)), desc = 'Processing') :
        for j in (range(0,len(combinationModel))) :
            for k in range(j+1,len(combinationModel)) :
                for l in range(k+1,len(combinationModel)) :
                    for n in range(l+1,len(combinationModel)) :
                        for o in range(n+1,len(combinationModel)) :
                            time.sleep(0.01)
                            FinalModelTrain = np.concatenate((combinationModel[j][1],combinationModel[l][1],combinationModel[k][1],combinationModel[n][1],combinationModel[o][1]),axis=1)
                            FinalModelTest = np.concatenate((combinationModel[j][2],combinationModel[l][2],combinationModel[k][2],combinationModel[n][2],combinationModel[o][2]),axis=1)
                            Final_Model = FinalModel[i].fit(FinalModelTrain, Y_train)
                            FinalModelPredict = Final_Model.predict(FinalModelTest)
                            FinalModelAccurancy = accuracy_score(Y_test,FinalModelPredict)
                            end = time.time() - start
                            SaveResultData = [model1[i],combinationModel[j][0],combinationModel[l][0],combinationModel[k][0],combinationModel[n][0],combinationModel[o][0],end,FinalModelAccurancy]
                            data_result2.loc[Num] = SaveResultData
                            Num += 1
                            MaxAccurancy = max(MaxAccurancy,FinalModelAccurancy)
                            start = time.time()

    data_result2.to_csv('Result/5 Combination Models.csv',index=False)
    print("Save the accuracy excel file for 5 machine learning combinations in the 'Result' Folder")
    print("max accuracy score" ,MaxAccurancy)
    print()
    MaxAccurancy = 0
    Num = 0
    print("↓↓↓ Start 7 Combination Model ↓↓↓")
    print()
    for i in tqdm(range(0,len(FinalModel)), desc = 'Processing'):
        for j in range(0,len(combinationModel)) :
            for k in range(j+1,len(combinationModel)) :
                for l in range(k+1,len(combinationModel)) :
                    for n in range(l+1,len(combinationModel)) :
                        for o in range(n+1,len(combinationModel)) :
                            for p in range(o+1,len(combinationModel)) :
                                for q in range(p+1,len(combinationModel)) :
                                    start = time.time()
                                    FinalModelTrain = np.concatenate((combinationModel[j][1],combinationModel[l][1],combinationModel[k][1],combinationModel[n][1],combinationModel[o][1],combinationModel[p][1],combinationModel[q][1]),axis=1)
                                    FinalModelTest = np.concatenate((combinationModel[j][2],combinationModel[l][2],combinationModel[k][2],combinationModel[n][2],combinationModel[o][2],combinationModel[p][2],combinationModel[q][2]),axis=1)
                                    Final_Model = FinalModel[i].fit(FinalModelTrain, Y_train)
                                    FinalModelPredict = Final_Model.predict(FinalModelTest)
                                    FinalModelAccurancy = accuracy_score(Y_test,FinalModelPredict)
                                    end = time.time() - start
                                    SaveResultData = [model1[i],combinationModel[j][0],combinationModel[l][0],combinationModel[k][0],combinationModel[n][0],combinationModel[o][0],combinationModel[p][0],combinationModel[q][0],end,FinalModelAccurancy]
                                    data_result3.loc[Num] = SaveResultData
                                    Num += 1
                                    MaxAccurancy = max(MaxAccurancy,FinalModelAccurancy)
                                    time.sleep(0.01)
    
    data_result3.to_csv('Result/7 Combination Models.csv',index=False)
    print("Save the accuracy excel file for 7 machine learning combinations in the 'Result' Folder")
    print("max accuracy score" ,MaxAccurancy)

if __name__=='__main__':
    X_train, Y_train, X_test, Y_test=ImportFeatureFile()
    IndependentModel(X_train, Y_train, X_test, Y_test)
    StackingModel(X_train, Y_train, X_test, Y_test,3)
