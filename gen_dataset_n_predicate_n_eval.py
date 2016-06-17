import os, nltk, pickle, re, time, requests, json
import numpy as np
import pandas as pd
from sklearn import mixture, cluster, linear_model, svm, neighbors, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from gensim.models import word2vec
import gensim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from collections import defaultdict
from collections import Counter
import math, operator
from nltk.corpus import stopwords
import string, pickle

def load_so(name):
    so_dict = pickle.load(open(name, "r+"))
    return so_dict

if __name__ == "__main__":      
    #Initial settings
    os.environ['TZ'] = 'US/Eastern'; time.tzset()
    
    #####################################################
    #print("\nget a sense of the so scores coming from w2v training")
    so_dict_stock = load_so("./data/so_scores/seman_oreint_dict_stock.p") 
    so_dict_employ = load_so("./data/so_scores/seman_oreint_dict_employ.p")
    so_dict_discrim = load_so("./data/so_scores/seman_oreint_dict_discrim.p")
    so_dict_educate = load_so("./data/so_scores/seman_oreint_dict_educate.p")
    so_dict_govern = load_so("./data/so_scores/seman_oreint_dict_govern.p")
    so_dict_polit = load_so("./data/so_scores/seman_oreint_dict_polit.p")

    #print("\ngenerate dataset from tweets and seman_oreint_dict.p") 
    POS_FEATS = True
    SOCIAL = True
    
    # 1) Get tweets
    tweets = []
    for stock_name in ["spx", "aapl", "msft", "xom", "jnj", "ge", "brk", "fb", "t", "amzn", "wfc"]:
        tweets.append(pd.read_pickle(stock_name + "_val"  + "_tweets.p"))
        tweets.append(pd.read_pickle(stock_name + "_tra"  + "_tweets.p"))
    tweets = pd.concat(tweets)  

    # 2) Go through each day and make prediction dataset
    spx_outcome = pd.read_csv("daily_price_changes.csv", header=0)
    tk = nltk.tokenize.casual.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    count_vect = CountVectorizer(tokenizer=lambda text: tk.tokenize(text), lowercase=False)
    punctuation = list(string.punctuation) + ["#", "*"]
    stop = stopwords.words('english')
    
    #choose to have POS SO features
    if(SOCIAL):
        dimensions = ["stock", "employ", "discrim", "educate", "govern", "polit"]
    else:
        dimensions = ["stock"]
    if(POS_FEATS):
        #pos = ["ADJ","ADP","ADV","CONJ","DET","NOUN","NUM","PRON","VERB","."]
        pos = ["CD", "DT", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "PRP", "RB", "RBR", "VB", "VBD", "VBG", "VBN", "VBZ", "WDT", "WP$"]
        feature_names = []
        for tag in pos:
            for dimension in dimensions:
                feature_names.append(tag + "_" + dimension)    
    else:
        feature_names = []
        for dimension in dimensions:
            feature_names.append("sum_so/#tweets" + "_" + dimension)

    pred_dataset_real = pd.DataFrame([], columns=feature_names + ['label (price change)'])
    pred_dataset_bool = pd.DataFrame([], columns=feature_names + ['label (price change up or down)'])    
    
    #for each day
    for yday in sorted(tweets["tm_yday"].unique()):
        if (not spx_outcome.loc[spx_outcome["yday"] == int(yday)].empty):
            price_change = float(spx_outcome.loc[spx_outcome["yday"] == int(yday)]["change in price"].values)
            tweet_count = len(tweets.loc[tweets["tm_yday"] == yday]["text"])

            tweets_text = tweets.loc[tweets["tm_yday"] == yday]["text"]
            tweets_tokenized = []
            for tweet in tweets_text:
                text = ' '.join(w for w in tk.tokenize(tweet) if w not in punctuation + stop)
                tweets_tokenized.append(text)
            
            counts = count_vect.fit_transform(tweets_tokenized)
            counts = counts.sum(axis=0)
            
            row = np.zeros(len(pred_dataset_real.columns))
            pred_dataset_real.loc[yday] = row
            pred_dataset_bool.loc[yday] = row
                
            if POS_FEATS: #get pos of word and put in right cell                
                for word in count_vect.vocabulary_.keys():
                    #stock
                    if so_dict_stock.has_key(word):
                        i = count_vect.vocabulary_.get(word)
                        count = int(np.array(counts)[0].tolist()[i])
                        so = so_dict_stock.get(word)
                        #tag = nltk.pos_tag([word], tagset="universal")[0][1]
                        tag = nltk.pos_tag([word])[0][1]
                        if tag in pos:
                            pred_dataset_real.loc[yday][tag + "_" + "stock"] += so*count
                            pred_dataset_bool.loc[yday][tag + "_" + "stock"] += so*count
                    if(SOCIAL):
                        #employ
                        if so_dict_employ.has_key(word):
                            i = count_vect.vocabulary_.get(word)
                            count = int(np.array(counts)[0].tolist()[i])
                            so = so_dict_employ.get(word)
                            #tag = nltk.pos_tag([word], tagset="universal")[0][1]
                            tag = nltk.pos_tag([word])[0][1]
                            if tag in pos:
                                pred_dataset_real.loc[yday][tag + "_" + "employ"] += so*count
                                pred_dataset_bool.loc[yday][tag + "_" + "employ"] += so*count
    
                        #discrim
                        if so_dict_discrim.has_key(word):
                            i = count_vect.vocabulary_.get(word)
                            count = int(np.array(counts)[0].tolist()[i])
                            so = so_dict_discrim.get(word)
                            #tag = nltk.pos_tag([word], tagset="universal")[0][1]
                            tag = nltk.pos_tag([word])[0][1]
                            if tag in pos:
                                pred_dataset_real.loc[yday][tag + "_" + "discrim"] += so*count
                                pred_dataset_bool.loc[yday][tag + "_" + "discrim"] += so*count
                                
                        #educate
                        if so_dict_educate.has_key(word):
                            i = count_vect.vocabulary_.get(word)
                            count = int(np.array(counts)[0].tolist()[i])
                            so = so_dict_educate.get(word)
                            #tag = nltk.pos_tag([word], tagset="universal")[0][1]
                            tag = nltk.pos_tag([word])[0][1]
                            if tag in pos:
                                pred_dataset_real.loc[yday][tag + "_" + "educate"] += so*count
                                pred_dataset_bool.loc[yday][tag + "_" + "educate"] += so*count
                                
                        #govern
                        if so_dict_govern.has_key(word):
                            i = count_vect.vocabulary_.get(word)
                            count = int(np.array(counts)[0].tolist()[i])
                            so = so_dict_govern.get(word)
                            #tag = nltk.pos_tag([word], tagset="universal")[0][1]
                            tag = nltk.pos_tag([word])[0][1]
                            if tag in pos:
                                pred_dataset_real.loc[yday][tag + "_" + "govern"] += so*count
                                pred_dataset_bool.loc[yday][tag + "_" + "govern"] += so*count
                                
                        #polit
                        if so_dict_polit.has_key(word):
                            i = count_vect.vocabulary_.get(word)
                            count = int(np.array(counts)[0].tolist()[i])
                            so = so_dict_polit.get(word)
                            #tag = nltk.pos_tag([word], tagset="universal")[0][1]
                            tag = nltk.pos_tag([word])[0][1]
                            if tag in pos:
                                pred_dataset_real.loc[yday][tag + "_" + "polit"] += so*count
                                pred_dataset_bool.loc[yday][tag + "_" + "polit"] += so*count            
                
                pred_dataset_real.loc[yday] = pred_dataset_real.loc[yday]/tweet_count
                pred_dataset_real.loc[yday]["label (price change)"] =  price_change
                pred_dataset_bool.loc[yday] = pred_dataset_bool.loc[yday]/tweet_count
                pred_dataset_bool.loc[yday]["label (price change up or down)"] =  indictor_fn(price_change)
                
            else:
                # feature_names.append("sum_so/#tweets" + "_" + dimension)
                
                #stock
                for word in count_vect.vocabulary_.keys():
                    if so_dict_stock.has_key(word):
                        i = count_vect.vocabulary_.get(word)
                        count = int(np.array(counts)[0].tolist()[i])
                        so = so_dict_stock.get(word)
                        pred_dataset_real.loc[yday]["sum_so/#tweets" + "_" + "stock"] += so*count
                        pred_dataset_bool.loc[yday]["sum_so/#tweets" + "_" + "stock"] += so*count
                        
                if(SOCIAL):
                    #employ
                    for word in count_vect.vocabulary_.keys():
                        if so_dict_employ.has_key(word):
                            i = count_vect.vocabulary_.get(word)
                            count = int(np.array(counts)[0].tolist()[i])
                            so = so_dict_employ.get(word)
                            pred_dataset_real.loc[yday]["sum_so/#tweets" + "_" + "employ"] += so*count
                            pred_dataset_bool.loc[yday]["sum_so/#tweets" + "_" + "employ"] += so*count
                            
                    #discrim
                    for word in count_vect.vocabulary_.keys():
                        if so_dict_discrim.has_key(word):
                            i = count_vect.vocabulary_.get(word)
                            count = int(np.array(counts)[0].tolist()[i])
                            so = so_dict_discrim.get(word)
                            pred_dataset_real.loc[yday]["sum_so/#tweets" + "_" + "discrim"] += so*count
                            pred_dataset_bool.loc[yday]["sum_so/#tweets" + "_" + "discrim"] += so*count
                            
                    #educate
                    for word in count_vect.vocabulary_.keys():
                        if so_dict_educate.has_key(word):
                            i = count_vect.vocabulary_.get(word)
                            count = int(np.array(counts)[0].tolist()[i])
                            so = so_dict_educate.get(word)
                            pred_dataset_real.loc[yday]["sum_so/#tweets" + "_" + "educate"] += so*count
                            pred_dataset_bool.loc[yday]["sum_so/#tweets" + "_" + "educate"] += so*count
                            
                    #govern
                    for word in count_vect.vocabulary_.keys():
                        if so_dict_govern.has_key(word):
                            i = count_vect.vocabulary_.get(word)
                            count = int(np.array(counts)[0].tolist()[i])
                            so = so_dict_govern.get(word)
                            pred_dataset_real.loc[yday]["sum_so/#tweets" + "_" + "govern"] += so*count
                            pred_dataset_bool.loc[yday]["sum_so/#tweets" + "_" + "govern"] += so*count
                            
                    #polit
                    for word in count_vect.vocabulary_.keys():
                        if so_dict_polit.has_key(word):
                            i = count_vect.vocabulary_.get(word)
                            count = int(np.array(counts)[0].tolist()[i])
                            so = so_dict_polit.get(word)
                            pred_dataset_real.loc[yday]["sum_so/#tweets" + "_" + "polit"] += so*count
                            pred_dataset_bool.loc[yday]["sum_so/#tweets" + "_" + "polit"] += so*count                 
                
                #print "\nSO/#tweet for day", yday, ":", so/tweet_count, "next day price change:", price_change
                #print " vocab count", len(count_vect.vocabulary_.keys()), "so:", so,  "tweet count", tweet_count
                pred_dataset_real.loc[yday] = pred_dataset_real.loc[yday]/tweet_count
                pred_dataset_real.loc[yday]["label (price change)"] =  price_change
                pred_dataset_bool.loc[yday] = pred_dataset_bool.loc[yday]/tweet_count
                pred_dataset_bool.loc[yday]["label (price change up or down)"] =  indictor_fn(price_change)
         
    #normalize
    if (True):
        #labels = pred_dataset_real['label (price change)']
        for feature in pred_dataset_real.columns:
            if feature != 'label (price change)':
                #MinMaxScaler   StandardScaler
                pred_dataset_real[feature] = preprocessing.StandardScaler().fit_transform( pred_dataset_real[feature].reshape(-1, 1) )
        for feature in pred_dataset_bool.columns:
            if feature != 'label (price change up or down)':
                pred_dataset_bool[feature] = preprocessing.StandardScaler().fit_transform( pred_dataset_bool[feature].reshape(-1, 1) )
    
    r = pred_dataset_real
    b = pred_dataset_bool
    ydays = sorted(r.index)
    midpoint = ydays[len(ydays)/2:][0]
    pred_dataset_real = r.loc[r.index < midpoint]
    pred_dataset_real_test = r.loc[r.index >= midpoint]
    pred_dataset_bool = b.loc[b.index < midpoint]
    pred_dataset_bool_test = b.loc[b.index >= midpoint]
     
    pred_dataset_real.to_csv("pred_dataset_real.csv")
    pred_dataset_real_test.to_csv("pred_dataset_real_test.csv")
    pred_dataset_bool.to_csv("pred_dataset_bool.csv")
    pred_dataset_bool_test.to_csv("pred_dataset_bool_test.csv")
    
    # 3) Predict and evaluate
    import evaluate_prediction_dataset2 as evaluate
    real = True
    evaluate.make_predictions(real, False, pred_dataset_real, "pred_dataset_real", pred_dataset_real_test)
    
    real = False
    evaluate.make_predictions(real, False, pred_dataset_bool, "pred_dataset_bool", pred_dataset_bool_test)  
    