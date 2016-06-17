import pandas as pd
import nltk, string

if __name__ == "__main__":
    # write tweets to a text file.  tweets are continiously concatenated together.  load all tweets for training w2v model
    
    tweets = []
    for stock_name in ["spx_old", "spx", "aapl", "msft", "xom", "jnj", "ge", "brk", "fb", "t", "amzn", "wfc"]:
        tweets.append(pd.read_pickle("./data/twitter/" + stock_name + "_val"  + "_tweets.p"))
        tweets.append(pd.read_pickle("./data/twitter/" + stock_name + "_tra"  + "_tweets.p"))
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
    f = open("./data/twitter/tweets_text.txt", "w+")
    #output tweet text
    tweets_text = ""
    for i, tweet in enumerate(tweets_tokenized):
        print i, "of", len(tweets_tokenized)
        text = ' '.join(w for w in tweet if w not in punctuation)
        f.write(text)

    f.close()