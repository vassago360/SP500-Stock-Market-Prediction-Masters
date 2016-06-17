from sklearn import datasets, preprocessing, cross_validation, metrics, tree, ensemble, feature_extraction, decomposition, feature_selection
from sklearn import naive_bayes, linear_model, svm, tree, neighbors, ensemble
from sklearn.utils.extmath import randomized_svd
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib2
import pickle, time, requests, json, csv, os, collections, multiprocessing, math
from random import randint

def same_sign(Y_test, predict_Y_test):
    if (Y_test * predict_Y_test) > 0:
        return True
    else:
        return False

def calc_wealth(Y_test, predict_Y_test):
    if same_sign(Y_test, predict_Y_test):
        if abs(Y_test) <= abs(predict_Y_test):
            wealth = abs(Y_test)
        else:
            wealth = abs(predict_Y_test)
    else:
        if abs(Y_test) <= abs(predict_Y_test):
            wealth = -(abs(predict_Y_test) + abs(Y_test))
        else:
            wealth = -abs(2*predict_Y_test)
    return wealth

class ma():
    def __init__(self,moving_average_data, real):
        self.real = real
        self.moving_average_data = moving_average_data
    def fit(self, X_train, Y_train):
        pass
    def score(self, X_test, Y_test):
        if self.moving_average_data.loc[X_test.index].values[0] < 0:
            val = -1
        elif self.moving_average_data.loc[X_test.index].values[0] == 0:
            val = 0
            return 0
        else:
            val = 1
        if int(Y_test) - val == 0:
            return 1
        else:
            return 0
    def predict(self, X_test):
        if self.real:
            return self.moving_average_data.loc[X_test.index].values #use the dates

def make_predictions(real, svd_bool, stock_market, sm_name, stock_market_test): #if real=True, do regression using prediction_dataset_real, otherwise do binary classification
    #moving_averages = pd.read_csv("moving_average_stock_data.csv", header=0, index_col=0)
    
    if real:
        names = ["RBF SVM R"]#, "5 moving average", "10 moving average"]#, "50 moving average"]#["Linear Regression", "RBF SVM R", "AdaBoost R", "5 moving average", "10 moving average"]#, "50 moving average"]
        classifiers = [
        svm.SVR(kernel='rbf'),
        #ma(moving_averages["5 moving average"], real),
        #ma(moving_averages["10 moving average"], real),
        ]
    else:
        names = ["Logistic Reg"]#, "5 moving average", "10 moving average"]#["Logistic Reg", "RBF SVM", "AdaBoost", "5 moving average", "10 moving average"]#, "50 moving average"]
        classifiers = [
        linear_model.LogisticRegression(),
        #ma(moving_averages["5 moving average"], real),
        #ma(moving_averages["10 moving average"], real),
         ]
    
    stock_market_train_data = stock_market.iloc[:,0:-1]
    stock_market_train_labels = stock_market.iloc[:,-1].values.ravel()
    train_X_train = stock_market_train_data
    train_Y_train = stock_market_train_labels

    start_offset = 2
    interval = 1
    offsets = range(start_offset,stock_market_test.shape[0],interval) #1 to stock_market_test.shape[0]-1 inclusive (so it's the row indices excluding [1] index) 
    
    accuracies = pd.DataFrame( np.zeros((len(classifiers),len(offsets))), index=names, columns=offsets )
    rmses = pd.DataFrame( np.zeros((len(classifiers),len(offsets))), index=names, columns=offsets )
    algs_predicts = pd.DataFrame( np.zeros((len(classifiers)+1,len(offsets))), index=names+["actual"], columns=offsets )
    if real:
        wealth = pd.DataFrame( np.zeros((len(classifiers)+1,len(offsets))), index=names+["sp500"], columns=offsets )
    else:
        wealth = pd.DataFrame( np.zeros((len(classifiers)+1,len(offsets))), index=names+["sp500 going long"], columns=offsets )
    
    last_offset = None
    for offset in offsets:
                
        #select the data using the offset to know how much to train and which instance to test
        stock_market_train_data = stock_market_test.iloc[0:offset,0:-1]
        stock_market_train_labels = stock_market_test.iloc[0:offset,-1].values.ravel()
        stock_market_test_data = stock_market_test.iloc[offset:offset+1,0:-1]
        stock_market_test_labels = stock_market_test.iloc[offset:offset+1,-1].values.ravel()
        
        X_train = pd.concat([train_X_train, stock_market_train_data])
        #print X_train
        Y_train = np.concatenate([train_Y_train, stock_market_train_labels])
        #print Y_train

        X_test = stock_market_test_data
        Y_test = stock_market_test_labels      
        
        for name, clf in zip(names, classifiers):        
            #train
            clf.fit(X_train, Y_train)
            
            if real:
                #predict
                if "moving average" in name:
                    predict_Y_test = clf.predict(stock_market_test_data) #need row names (dates)
                elif (clf.__class__.__name__ == 'KNeighborsRegressor') and (clf.n_neighbors > X_train.shape[0]):
                    predict_Y_test = np.zeros(len(Y_test))
                else:
                    predict_Y_test = clf.predict(X_test)
                #evaluate
                clf_rmse = math.sqrt(metrics.mean_squared_error(Y_test, predict_Y_test)) #rmse
                rmses.loc[name, offset] = clf_rmse
            else:
                #predict and evaluate
                clf_acc = clf.score(X_test, Y_test) #accuracy
                accuracies.loc[name, offset] = clf_acc
            
            if real:
                algs_predicts.loc[name, offset] = predict_Y_test
                algs_predicts.loc["actual", offset] = Y_test   
                if last_offset != None:
                    wealth.loc[name, offset] = calc_wealth(Y_test, predict_Y_test) + wealth.loc[name, last_offset]
                    wealth.loc["sp500", offset] = Y_test + wealth.loc["sp500", last_offset]
                else:
                    wealth.loc[name, offset] = calc_wealth(Y_test, predict_Y_test)
                    wealth.loc["sp500", offset] = Y_test
            else:
                if last_offset != None:
                    wealth.loc[name, offset] = (2*clf_acc)-1 + wealth.loc[name, last_offset]
                    wealth.loc["sp500 going long", offset] = Y_test + wealth.loc["sp500 going long", last_offset]
                else:
                    wealth.loc[name, offset] = (2*clf_acc)-1  
                    wealth.loc["sp500 going long", offset] = Y_test
        
        last_offset = offset
    
    if real:
        rmses.to_csv("eval_rmse_" + sm_name + ".csv")
        algs_predicts.to_csv("eval_predictions_" + sm_name + ".csv")
        #wealth.to_csv("eval_wealth_" + sm_name + ".csv")
        #summary = pd.concat([rmses.mean(axis=1), wealth.iloc[:,-1]],axis=1)
        #summary.columns = ["rmse mean", "last day accum wealth"]
        summary = rmses.mean(axis=1)
        summary.columns = ["rmse mean"]
    else:
        accuracies.to_csv("eval_acc_" + sm_name + ".csv")
        #   algs_predicts.to_csv("eval_predictions_" + sm_name + ".csv")  <--- would be ugly to graph (fluctuate a lot) so not doing it
        wealth.to_csv("eval_wealth_" + sm_name + ".csv")
        summary = pd.concat([accuracies.mean(axis=1), wealth.iloc[:,-1]],axis=1)
        summary.columns = ["acc mean", "last day accum wealth"]
        
    print summary


if __name__ == "__main__":
    real = True
    prediction_dataset = pd.read_csv("prediction_dataset_real.csv", header=0, index_col=0)
    make_predictions(real, False, prediction_dataset, "prediction_dataset_real")
    
    real = False
    prediction_dataset = pd.read_csv("prediction_dataset_bool.csv", header=0, index_col=0)
    make_predictions(real, False, prediction_dataset, "prediction_dataset_bool")