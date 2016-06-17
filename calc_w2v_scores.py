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

def indictor_fn(price_diff_prev_day):
    if price_diff_prev_day < 0:
        return -1 #stock price went down
    else:
        return 1#stock price went up   

def get_sentiment(tweets_text):
    sentiment_avg = 0
    data = [] # [{"text": "I love Titanic.", "query": "Titanic", "id": 1234},...]
    #for i in tweets_text.index:
    #    data.append({"text": tweets_text.loc[tweets_text.index[i]], "i":i})
    for tweet_text in tweets_text:
        data.append({"text": tweet_text})
    sentiment_results = get_api_sentiment({"data": data})
    for sentiment_result in sentiment_results:
        sentiment_avg += (sentiment_result["polarity"] - 2)
    sentiment_avg = sentiment_avg/float(len(tweets_text))
    return sentiment_avg

def get_api_sentiment(payload):
    url = "http://www.sentiment140.com/api/bulkClassifyJson?appid=sthornhi@ucsc.edu"
    headers = {'content-type': 'application/json'}
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
    except:
        print "sleeping (sentiment 140)..."
        print "data size ", len(payload['data'])
        time.sleep(900)
        response = requests.post(url, data=json.dumps(payload), headers=headers)
    return response.json()['data']

def load_so(name):
    #so_dict = pickle.load(open("seman_oreint_dict_lr_1xanto_oplex100_60epoch.p", "r+"))
    so_dict = pickle.load(open(name, "r+"))
    #verage = 0
    #poscount = 0
    #negcount = 0
    #for word in so_dict:
    #    average += so_dict.get(word)
    #    if so_dict.get(word) > 0:
    #        poscount += 1
    #    else:
    #        negcount += 1
    #average = average/len(so_dict.keys())
    #print "\n\n", name
    #print "average so:", average
    #print "poscount", poscount
    #print "negcount", negcount
    return so_dict

if __name__ == "__main__":      
    #Initial settings
    os.environ['TZ'] = 'US/Eastern'; time.tzset()
    
    #####################################################
    #print("\nget a sense of the so scores coming from w2v training")
    so_dict_stock = load_so("seman_oreint_dict_stock.p") 
    so_dict_employ = load_so("seman_oreint_dict_employ.p")
    so_dict_discrim = load_so("seman_oreint_dict_discrim.p")
    so_dict_educate = load_so("seman_oreint_dict_educate.p")
    so_dict_govern = load_so("seman_oreint_dict_govern.p")
    so_dict_polit = load_so("seman_oreint_dict_polit.p")

    #print("\ngenerate dataset from tweets and seman_oreint_dict.p") 
    POS_FEATS = True
    SOCIAL = True
    SENTIMENT = False #if use i need to run line 101
    
    # 1) Get tweets
    tweets = []
    for stock_name in ["spx"]: #, "aapl", "msft", "xom", "jnj", "ge", "brk", "fb", "t", "amzn", "wfc"]:  #"spx_old", 
        tweets.append(pd.read_pickle(stock_name + "_val"  + "_tweets.p"))
        tweets.append(pd.read_pickle(stock_name + "_tra"  + "_tweets.p"))
    tweets = pd.concat(tweets)  
    
    if False:
        daily_agg_sentiment = pd.DataFrame([], columns=["agg sentiment"])
        for yday in sorted(tweets["tm_yday"].unique()):
            tweets_text = tweets.loc[tweets["tm_yday"] == yday]["text"]
            sentiment = get_sentiment(tweets_text)
            daily_agg_sentiment.loc[yday] = np.array([sentiment])
        #if(CREATE_TEST): #get sentiment
        #    daily_agg_sentiment.to_csv("daily_agg_sentiment_test_data.csv")
        #else:
        #    daily_agg_sentiment.to_csv("daily_agg_sentiment_train_data.csv")

    # 2) Go through each day and make prediction dataset
    spx_outcome = pd.read_csv("daily_price_changes.csv", header=0)
    #if(CREATE_TEST):
    #    daily_agg_sentiment = pd.read_csv("daily_agg_sentiment_test_data.csv", header=0, index_col=0)
    #else:
    #    daily_agg_sentiment = pd.read_csv("daily_agg_sentiment_train_data.csv", header=0, index_col=0)
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
    if(SENTIMENT):
        feature_names.append('sentiment')
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
         
                if(SENTIMENT):
                    pred_dataset_real.loc[yday]["sentiment"] = daily_agg_sentiment.loc[int(yday)]
                    pred_dataset_bool.loc[yday]["sentiment"] = daily_agg_sentiment.loc[int(yday)]
                
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
                
                if(SENTIMENT):
                    pred_dataset_real.loc[yday]["sentiment"] = daily_agg_sentiment.loc[int(yday)]
                    pred_dataset_bool.loc[yday]["sentiment"] = daily_agg_sentiment.loc[int(yday)]                  
                
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
    #import evaluate_prediction_dataset as evaluate
    #real = True
    #evaluate.make_predictions(real, False, pred_dataset_real, "pred_dataset_real")
    
    #real = False
    #evaluate.make_predictions(real, False, pred_dataset_bool, "pred_dataset_bool")  
    import evaluate_prediction_dataset2 as evaluate
    real = True
    evaluate.make_predictions(real, False, pred_dataset_real, "pred_dataset_real", pred_dataset_real_test)
    
    real = False
    evaluate.make_predictions(real, False, pred_dataset_bool, "pred_dataset_bool", pred_dataset_bool_test)  
    
    
    jj
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ######################################################
    # write tweets to a text file.  tweets are continiously concatenated together.  load all tweets for training w2v model
    print("\nwrite tweets to text file")
    tweets = []
    for stock_name in ["spx_old", "spx", "aapl", "msft", "xom", "jnj", "ge", "brk", "fb", "t", "amzn", "wfc"]:
        tweets.append(pd.read_pickle(stock_name + "_val"  + "_tweets.p"))
        tweets.append(pd.read_pickle(stock_name + "_tra"  + "_tweets.p"))
    tweets = pd.concat(tweets)
    
    tweets_text = []
    for tweet in tweets["text"]:
        tweets_text.append(tweet)
    tweets_text = list(set(tweets_text))
    
    tweets_tokenized = []
    tk = nltk.tokenize.casual.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True) 
    for tweet in tweets_text:
        tweets_tokenized.append(tk.tokenize(tweet))  
    
    punctuation = list(string.punctuation) + ["#", "*"]
    f = open("tweets_text.txt", "w+")
    #output tweet text
    tweets_text = ""
    for i, tweet in enumerate(tweets_tokenized):
        print i, "of", len(tweets_tokenized)
        #text = ' '.join(tweet).translate(None, string.punctuation).replace("  ", " ")
        text = ' '.join(w for w in tweet if w not in punctuation)
        f.write(text)

    f.close()
    
    
    ######################################################
    print "\ntrain gensim w2v model.  loading twitter data and train w2v..."
    if (False):
        #load all tweets for training w2v model
        tweets = []
        for stock_name in ["spx_old", "spx", "aapl", "msft", "xom", "jnj", "ge", "brk", "fb", "t", "amzn", "wfc"]:
            tweets.append(pd.read_pickle(stock_name + "_val"  + "_tweets.p"))
            tweets.append(pd.read_pickle(stock_name + "_tra"  + "_tweets.p"))
        tweets = pd.concat(tweets)
        
        
        #train w2v model
        tweets_text = []
        for tweet in tweets["text"]:
            tweets_text.append(tweet)
        tweets_text = list(set(tweets_text))
        
        tweets_tokenized = []
        tk = nltk.tokenize.casual.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True) 
        for tweet in tweets_text:
            tweets_tokenized.append(tk.tokenize(tweet))  
            
        model = gensim.models.Word2Vec(tweets_tokenized, min_count=2, workers=5)
        model.save('my_w2v_model')
    
    #load just the train tweets for further analysis
    tweets = []
    for stock_name in ["spx_old", "spx"]: #["spx_old", "spx", "aapl", "msft", "xom", "jnj", "ge", "brk", "fb", "t", "amzn", "wfc"]: #
        tweets.append(pd.read_pickle(stock_name + "_tra"  + "_tweets.p"))
    tweets = pd.concat(tweets)    
    
    
    ######################################################
    print "\ncalculate Semantic Orientation..."
    #get how the stock market did 
    w2vModel = gensim.models.Word2Vec.load('my_w2v_model')
    spx_outcome = pd.read_csv("daily_price_changes.csv", header=0)
    polarity_dict = pd.read_csv("10_stock.txt", header=None, names=["pos", "neg"])
    neg_terms = polarity_dict["neg"]
    pos_terms = polarity_dict["pos"]
    
    #define SO, get aggregate value for a day
    tk = nltk.tokenize.casual.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)  #<--- twitter tokenizer
    count_vect = CountVectorizer(tokenizer=lambda text: tk.tokenize(text), lowercase=False)

    #for yday in sorted(tweets["tm_yday"].unique()):
    #    count_vect.fit_transform(tweets.loc[tweets["tm_yday"] == yday]["text"]) #text for that day    
    #    print yday, len(count_vect.vocabulary_.keys())
    prediction_dataset_real = pd.DataFrame([], columns=['sum_of_bullish_dists/tweetcount', 'sum_of_bearish_dists/tweetcount', 'label (price change)'])
    prediction_dataset_bool = pd.DataFrame([], columns=['sum_of_bullish_dists/tweetcount', 'sum_of_bearish_dists/tweetcount', 'label (price change up or down)'])
    
    for yday in sorted(tweets["tm_yday"].unique()):
        counts = count_vect.fit_transform(tweets.loc[tweets["tm_yday"] == yday]["text"]) #text for that day
        counts = count_vect.fit_transform(tweets_tokenized)
        counts = counts.sum(axis=0)

        #count_vect_vocab
        sum_of_bullish_dists = 0
        sum_of_bearish_dists = 0
        total_pos_word_count = 0
        total_neg_word_count = 0
        todays_pos_terms = []
        todays_neg_terms = []
        for word in pos_terms:
            if word in count_vect.vocabulary_.keys():
                if word in w2vModel.vocab:
                    todays_pos_terms.append(word)
                    i = count_vect.vocabulary_.get(word)
                    count = int(np.array(counts)[0].tolist()[i])
                    weight = count
                    sum_of_bullish_dists += weight
                    total_pos_word_count += count
                    ###
                    
                else:
                    print word, "is not in w2v model"
            #else:
            #    print word, "not found for day", yday
        
        for word in neg_terms:
            if word in count_vect.vocabulary_.keys():
                if word in w2vModel.vocab:
                    todays_neg_terms.append(word)
                    i = count_vect.vocabulary_.get(word)
                    count = int(np.array(counts)[0].tolist()[i])
                    weight = count
                    sum_of_bearish_dists += weight
                    total_neg_word_count += count
                else:
                    print word, "is not in w2v model"
            #else:
            #    print word, "not found for day", yday
            
        pos_compare_terms = []
        for word in pos_terms['w1']:
            if word in w2vModel.vocab:
                pos_compare_terms.append(word)
        neg_compare_terms = []
        for word in neg_terms['w1']:
            if word in w2vModel.vocab:
                neg_compare_terms.append(word)      
                  
        for term, score in w2vModel.most_similar(positive=pos_compare_terms, negative=neg_compare_terms, topn=10):
            if word in count_vect.vocabulary_.keys():
                count = count_vect.vocabulary_[word]
                weight = count*score
                sum_of_bullish_dists += weight
                total_pos_word_count += count
        for term, score in w2vModel.most_similar(positive=neg_compare_terms, negative=pos_compare_terms, topn=10):
            if word in count_vect.vocabulary_.keys():
                count = count_vect.vocabulary_[word]
                weight = count*score
                sum_of_bearish_dists += weight
                total_neg_word_count += count
 
               
        #so = sum_of_bullish_dists/(total_pos_word_count+1) + sum_of_bearish_dists/(total_neg_word_count+1)
        so = sum_of_bullish_dists + sum_of_bearish_dists
        if (not spx_outcome.loc[spx_outcome["yday"] == int(yday)].empty): #and (int(yday) > 109):
            price_change = float(spx_outcome.loc[spx_outcome["yday"] == int(yday)]["stock_price_real"].values)
            tweet_count = len(tweets.loc[tweets["tm_yday"] == yday]["text"])
            print "\nSO for day", yday, ":", so, "next day price change:", price_change, "; so = ", sum_of_bullish_dists/(total_pos_word_count+1), "(c:", total_pos_word_count,")", "+", sum_of_bearish_dists/(total_neg_word_count+1), "(c:", total_neg_word_count,")"
            print " vocab count", len(count_vect.vocabulary_.keys()),  "tweet count", tweet_count, "pos words count", total_pos_word_count, "neg words count", total_neg_word_count
            prediction_dataset_real.loc[yday] = np.array([sum_of_bullish_dists/tweet_count, sum_of_bearish_dists/tweet_count, price_change])
            prediction_dataset_bool.loc[yday] = np.array([sum_of_bullish_dists/tweet_count, sum_of_bearish_dists/tweet_count, indictor_fn(price_change)])

    #normalize
    if (True):
        labels = prediction_dataset_real['label (price change)']
        prediction_dataset_real = (prediction_dataset_real - prediction_dataset_real.mean()) / (prediction_dataset_real.max() - prediction_dataset_real.min())
        prediction_dataset_real['label (price change)'] = labels
        
        labels = prediction_dataset_bool['label (price change up or down)']
        prediction_dataset_bool = (prediction_dataset_bool - prediction_dataset_bool.mean()) / (prediction_dataset_bool.max() - prediction_dataset_bool.min())
        prediction_dataset_bool['label (price change up or down)'] = labels

    prediction_dataset_real.to_csv("prediction_dataset_real.csv")
    prediction_dataset_bool.to_csv("prediction_dataset_bool.csv")
    
    import evaluate_prediction_dataset as evaluate
    real = True
    prediction_dataset = pd.read_csv("prediction_dataset_real.csv", header=0, index_col=0)
    evaluate.make_predictions(real, False, prediction_dataset, "prediction_dataset_real")
    
    real = False
    prediction_dataset = pd.read_csv("prediction_dataset_bool.csv", header=0, index_col=0)
    evaluate.make_predictions(real, False, prediction_dataset, "prediction_dataset_bool")
    
    ######################################################
    print "\ngenerate prediction dataset (average w2v representation) to predict next day (10 min eta)"
    w2vModel = gensim.models.Word2Vec.load('my_w2v_model')
    
    punctuation = list(string.punctuation) + ["#", "*"]
    stop = stopwords.words('english') + punctuation + ["#"]
    
    tk = nltk.tokenize.casual.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)  #<--- twitter tokenizer
    count_vect = CountVectorizer(tokenizer=lambda text: tk.tokenize(text), lowercase=False)
    
    spx_outcome = pd.read_csv("daily_price_changes.csv", header=0)
    prediction_dataset_real = pd.DataFrame([], columns=range(100) + ["label (price change)"])
    prediction_dataset_bool = pd.DataFrame([], columns=range(100) + ["label (price change up or down)"])
    
    for yday in sorted(tweets["tm_yday"].unique()):
        #print yday
        tweets_for_the_day = tweets.loc[tweets["tm_yday"] == yday]["text"]
        X_counts = count_vect.fit_transform(tweets_for_the_day) #text for that day
        #X_tfidf = TfidfTransformer().fit_transform(X_counts)
        w2v_day_vec = np.array([0.0]*100)
        for tweet in tweets_for_the_day:
            tweet_tokenized = tk.tokenize(tweet)
            for term in tweet_tokenized:
                if (not (term in stop)) and (not bool(re.search("^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", term))) and (not bool(re.search("^[+-]?(\d+(,\d*)?|,\d+)$", term))):
                    if term in w2vModel.vocab:    
                        w2v_word_vec = normalize(w2vModel[term].reshape(1, -1)) #ensure unit length
                        #tfidf_word_weight = tfidf_word_weights[i, tfidf_vocab.get(word)]
                        w2v_day_vec += w2v_word_vec.flatten()
        if not spx_outcome.loc[spx_outcome["yday"] == int(yday)].empty:
            price_change = float(spx_outcome.loc[spx_outcome["yday"] == int(yday)]["stock_price_real"].values)
            prediction_dataset_real.loc[yday] = np.append(normalize(w2v_day_vec.reshape(1, -1)).flatten(), price_change)
            prediction_dataset_bool.loc[yday] = np.append(normalize(w2v_day_vec.reshape(1, -1)).flatten(), indictor_fn(price_change))
                    
    prediction_dataset_real.to_csv("prediction_dataset_real.csv")
    prediction_dataset_bool.to_csv("prediction_dataset_bool.csv")
    
    ######################################################
    print "\nwords with highest PMI to 'bullish' & 'bearish' from the tweets (5 min eta)"
    # 1) Get tweets
    tweets = []
    for stock_name in ["spx_old", "spx", "aapl", "msft", "xom", "jnj", "ge", "brk", "fb", "t", "amzn", "wfc"]:
        tweets.append(pd.read_pickle(stock_name + "_val"  + "_tweets.p"))
        tweets.append(pd.read_pickle(stock_name + "_tra"  + "_tweets.p"))
    tweets = pd.concat(tweets)
    
    
    tweets_text = []
    for tweet in tweets["text"]:
        tweets_text.append(tweet)
    tweets_text = list(set(tweets_text))
    print "tweets_text count:", len(tweets_text)

    #build a co-occurrence matrix com such that com[x][y] contains the number of times the term x has been seen in the same tweet as the term y
    com = defaultdict(lambda : defaultdict(int))
    tweets_tokenized = []
    tk = nltk.tokenize.casual.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True) 
    for tweet in tweets_text:
        terms_only = tk.tokenize(tweet)
        tweets_tokenized.append(terms_only)
        # Build co-occurrence matrix
        for i in range(len(terms_only)-1):            
            for j in range(i+1, len(terms_only)):
                w1, w2 = sorted([terms_only[i], terms_only[j]]) 
                if w1 != w2:
                    com[w1][w2] += 1
                    com[w2][w1] += 1

    #document frequency
    print "now doc freq"
    document_frequencies = Counter()
    #tweets_tokenized = [tk.tokenize(tweet) for tweet in tweets_text]
    map(document_frequencies.update, (tweet_tokenized for tweet_tokenized in tweets_tokenized))

    #probabilities
    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + ["#"]
    
    p_t = {}
    p_t_com = defaultdict(lambda : defaultdict(int))
    
    n_docs = float(len(tweets_text))
    for term, n in document_frequencies.items():
        if (not (term in stop)) and (not bool(re.search("^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", term))) and (not bool(re.search("^[+-]?(\d+(,\d*)?|,\d+)$", term))) and (n > 4):
            p_t[term] = n / n_docs
            for t2 in com[term]:
                if (not (t2 in stop)) and (not bool(re.search("^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", t2))) and (not bool(re.search("^[+-]?(\d+(,\d*)?|,\d+)$", t2))) and (n > 4):
                    if com[term][t2] > 1:
                        p_t_com[term][t2] = com[term][t2] / n_docs
                        p_t_com[t2][term] = com[t2][term] / n_docs
                    else:
                        p_t_com[term][t2] = 0
                        p_t_com[t2][term] = 0                          
            
    #two vocabularies for positive and negative terms     
    
    polarity_dict = pd.read_csv("all_stock_words.txt", header=None, names=["pos", "neg"])
    neg_terms = polarity_dict["neg"]
    pos_terms = polarity_dict["pos"]
    
    positive_vocab = pos_terms.tolist()
    negative_vocab = neg_terms.tolist()
    
    #semantic orientation
    pmi = defaultdict(lambda : defaultdict(int))
    for t1 in p_t:
        for t2 in com[t1]:
            if p_t.has_key(t2):
                denom = p_t[t1] * p_t[t2]
                if p_t_com[t1][t2] != 0:
                    pmi[t1][t2] = math.log(p_t_com[t1][t2] / denom, 2)
                else:
                    pmi[t1][t2] = 0
     
    semantic_orientation = {}
    for term, n in p_t.items():
        positive_assoc = 0
        for w in positive_vocab:
            if pmi[term].has_key(w):
                positive_assoc += pmi[term][w]
        negative_assoc = 0
        for w in negative_vocab:
            if pmi[term].has_key(w):
                negative_assoc += pmi[term][w]
        #positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
        #negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
        semantic_orientation[term] = positive_assoc - negative_assoc
        
    pickle.dump(semantic_orientation, open("seman_oreint_dict_stock_pmi.p", "w+"))
    
    #print out the semantic orientation for some terms
    semantic_sorted = sorted(semantic_orientation.items(), 
                             key=operator.itemgetter(1), 
                             reverse=True)
    top_pos = semantic_sorted[:500]
    top_neg = semantic_sorted[-500:]
    top_neg.reverse()
     
    print("\ntop_pos:")
    for term, score in top_pos:
        positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
        negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
        print term, score, "=", positive_assoc, "-", negative_assoc
    print("\ntop_neg:")
    for term, score in top_neg:
        positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
        negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
        print term, score, "=", positive_assoc, "-", negative_assoc
        
    #normalize scores
    total_pos_score = 0
    for term, score in top_pos:
        total_pos_score += score
    total_neg_score = 0
    for term, score in top_neg:
        total_neg_score += score
    for i, term_score in enumerate(top_pos):
        term, score = term_score
        top_pos[i] = (term, score/total_pos_score)
    for i, term_score in enumerate(top_neg):
        term, score = term_score
        top_neg[i] = (term, score/total_neg_score)
    
    ######################################################
    print "\nlook at word distances of words to bullish/bearish..."
    w2vModel = gensim.models.Word2Vec.load('my_w2v_model')
    bullish_vec = w2vModel["bullish"]
    bearish_vec = w2vModel["bearish"]
    
    tk = nltk.tokenize.casual.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)  #<--- twitter tokenizer
    count_vect = CountVectorizer(tokenizer=lambda text: tk.tokenize(text), lowercase=False)
    count_vect.fit_transform(tweets["text"])
    count_vect.vocabulary_
    
    word_distances = pd.DataFrame([], columns=["in vocab", "bullish sim", "bearish sim"]) #index="word"
    for word in ["gc", "smh", "es", "bullish", "rallies", "intermediate", "nasty", "bearish"]:#count_vect.vocabulary_:
        #store word, cossim
        if word in w2vModel.vocab:
            #word_vec = w2vModel.syn0norm[w2vModel.vocab[word].index]
            word_vec = w2vModel[word]
            bullish_sim = metrics.pairwise.cosine_similarity(word_vec.reshape(1, -1), bullish_vec.reshape(1, -1))[0][0]
            bearish_sim = metrics.pairwise.cosine_similarity(word_vec.reshape(1, -1), bearish_vec.reshape(1, -1))[0][0]
            word_distances.loc[word] = [True, bullish_sim, bearish_sim]
        else:
            word_distances.loc[word] = [False, 0, 0]
    word_distances.to_csv("word_distances.csv")
    
    print word_distances.sort_values(by="bullish sim")
    print word_distances.sort_values(by="bearish sim")
    



    ######################################################
    print "\nwhat's most similar to -bullish_vec ??"
    w2vModel = gensim.models.Word2Vec.load('my_w2v_model')
    bullish_vec = w2vModel["bullish"]
    bearish_vec = w2vModel["bearish"]
    
    w2vModel.most_similar(positive=['bullish'], negative=['bearish'])
    
    neigh = neighbors.NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute').fit(w2vModel.syn0norm)
    distances, indices = neigh.kneighbors((-bullish_vec).reshape(1, -1))
    negbullish_nn_indices = indices[0]
    distances, indices = neigh.kneighbors((-bearish_vec).reshape(1, -1))
    negbearish_nn_indices = indices[0]
    for word in w2vModel.vocab.keys():
        if w2vModel.vocab[word].index in negbullish_nn_indices:
            print "  nn of -bullish:", word
    print "\n"
    for word in w2vModel.vocab.keys():
        if w2vModel.vocab[word].index in negbearish_nn_indices:
            print "  nn of -bearish:", word
    bullish_sim = metrics.pairwise.cosine_similarity((-bullish_vec).reshape(1, -1), bullish_vec.reshape(1, -1))[0][0] # = -1
    
    

        



        

    


