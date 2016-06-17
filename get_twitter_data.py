import twitter, pickle, time, requests, json, os, collections, multiprocessing, re
from sklearn import cross_validation
import numpy as np
import pandas as pd

def _twttr_call(return_dict, api, api_func, args):
    # consumer_key , consumer_secret , access_token_key , access_token_secret
    accounts = []
    accounts.append( ['vtxeKRdnnZViFMLdPQ6Wn8g4a','qtRHVAC6vu4D3DX3ZLlnNxsYNvWUplgSbMwoq1F4ekqswhDNqr','572824457-rUpwiWuv0RJCZgBBiSZYoRCDLhCjNgtmjQcQW0mG','wQepR6Hdi3DN3nKtXCiAvk3Kj9pKrR53aZWtBdttioMVN'] )
    accounts.append( ['GT4qPbnixWRSUcKS1VdoDtuU6', 'yjEfJDbEYj16v31OzyvZLvPRzkjAT4AKdBMpCnhU05joTqsMLs', '572824457-2pqYz4P91V0FROENm4RrB1IVselD5WRPOEuHchIU', 'KItjNr1dOgRKLugFaYqheAgBpOMcnLjrif7GCZCDZAlsH'] )
    accounts.append( ['SrhvxVuuiB42AG6uuowU8HUDJ', 'nIDqCJek6tpquO8fmo6wqkgojY6gCPtL6lFPQqeNOFBYJT3jSz', '4878561863-e68PDfuGATWbLHn0E0XZsQB286cEzZUOzJMq577', 'D20Tj8azLfco0p5HBDKEre8ffYCSxHurWccAuEGh3fwgC'] )
    # first pass (use same credentials)
    try:
        return_dict["_twttr_call"] = api_func(**args)
        return
    except twitter.error.TwitterError as e:
        if e.args  == ([{u'message': u'Rate limit exceeded', u'code': 88}],):
            pass
        elif e.args[0] == u'Not authorized.':
            print "  not authorized..."
            return_dict["_twttr_call"] = None
            return
        else:  
            print "  ", e.args, args, api_func #unexpected error
            return_dict["_twttr_call"] = []  
            return
    # second pass (change credentials)
    for account in accounts:
        if account[0] == api._consumer_key:
            continue
        else:
            #print "  switching accounts..."
            api.SetCredentials(consumer_key=account[0], consumer_secret=account[1], access_token_key=account[2], access_token_secret=account[3])
            try:
                return_dict["_twttr_call"] = api_func(**args)
                return
            except twitter.error.TwitterError as e:
                if e.args  == ([{u'message': u'Rate limit exceeded', u'code': 88}],):
                    continue #this account is also reached rate limit
                elif e.args[0] == u'Not authorized.':
                    print "  not authorized..."
                    return_dict["_twttr_call"] = None
                    return
                else:  
                    print "   ", e.args, args, api_func #unexpected error
                    return_dict["_twttr_call"] = []  
                    return
    # no luck, must sleep
    return_dict["sleeping"] = True
    return

def twttr_call(timeout, api, api_func, args):
    if timeout:
        while True:
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            return_dict["_twttr_call"] = []
            return_dict["sleeping"] = False
            p = multiprocessing.Process(target=_twttr_call, name="_twttr_call", args=(return_dict, api, api_func, args))
            p.start()
            p.join(timeout)
            if p.is_alive():
                print "  terminating"
                p.terminate()
                p.join()
                break
            elif return_dict["sleeping"]:
                time.sleep(900)
                print "  sleeping..."
            else:
                #got something :)
                break
    else:
        while True:
            return_dict = {}
            return_dict["_twttr_call"] = []
            return_dict["sleeping"] = False
            _twttr_call(return_dict, api, api_func, args)
            if return_dict["sleeping"]:
                print "  sleeping..."
                time.sleep(900)
            else:
                #got something :)
                break   
    return return_dict["_twttr_call"]

def get_status_text(status):
    if status.GetRetweeted_status():
        text = status.GetRetweeted_status().GetText()
    else:
        text = status.GetText()
    text = text.encode('ascii',errors='ignore')
    text = re.sub('@\S+','*ACCOUNT*', text)
    text = re.sub('(\*ACCOUNT\*(\s)*)+','*ACCOUNT* ', text)
    text = re.sub('https\S+','*LINK*', text)
    text = re.sub('(\*LINK\*(\s)*)+','*LINK* ', text)
    return text

def collect_tweets(stock_name, search_queries):
    #collect tweets for stock_name
    try:
        tweets_val = pd.read_pickle(stock_name + "_val" + "_tweets.p")
        val_since_id = max(tweets_val.index)
        tweets_tra = pd.read_pickle(stock_name + "_tra" + "_tweets.p")
        tra_since_id = max(tweets_tra.index)
        since_id = max([val_since_id, tra_since_id])
    except IOError:
        tweets_val = pd.DataFrame([], columns=["sn", "text", "epoch secs", "tm_yday", "tm_hour", "tm_min"]) #index="id"
        tweets_tra = pd.DataFrame([], columns=["sn", "text", "epoch secs", "tm_yday", "tm_hour", "tm_min"]) #index="id"
        since_id = None
    new_tweets = pd.DataFrame([], columns=["sn", "text", "epoch secs", "tm_yday", "tm_hour", "tm_min"])
    statuses = query_twitter(search_queries, since_id)    
    for s in statuses:
        tweet_id = s.GetId() 
        screen_name = s.GetUser().GetScreenName()
        text = get_status_text(s)
        epoch_secs = s.GetCreatedAtInSeconds()
        tm_yday = time.localtime(s.GetCreatedAtInSeconds()).tm_yday
        tm_hour = time.localtime(s.GetCreatedAtInSeconds()).tm_hour
        tm_min = time.localtime(s.GetCreatedAtInSeconds()).tm_min
        new_tweets.loc[tweet_id] = [screen_name, text, epoch_secs, tm_yday, tm_hour, tm_min]
    #tweets.to_pickle(stock_name + "_tweets.p")
    [new_tweets_tra, new_tweets_val] = cross_validation.train_test_split(new_tweets, test_size=.3)
    pd.concat([tweets_val, new_tweets_val]).to_pickle(stock_name + "_val" + "_tweets.p")
    pd.concat([tweets_tra, new_tweets_tra]).to_pickle(stock_name + "_tra" + "_tweets.p")
    
    
def query_twitter(search_queries, since_id=None): #don't get tweet earlier than since_id
    #initialize api
    api = twitter.Api(consumer_key='vtxeKRdnnZViFMLdPQ6Wn8g4a', consumer_secret='qtRHVAC6vu4D3DX3ZLlnNxsYNvWUplgSbMwoq1F4ekqswhDNqr', access_token_key='572824457-rUpwiWuv0RJCZgBBiSZYoRCDLhCjNgtmjQcQW0mG', access_token_secret='wQepR6Hdi3DN3nKtXCiAvk3Kj9pKrR53aZWtBdttioMVN')
    #query twitter till all results are collected
    total_search_results = []
    for search_query in search_queries:
        search_query_results = []
        max_id = None #don't get tweet later than max_id
        while True:
            search_results = twttr_call(0, api, api.GetSearch, {"term":search_query, "since_id":since_id, "max_id":max_id, "lang":"en", "count":200})
            if len(search_results) ==  0:
                break
            if search_results[-1].GetId() == max_id:
                break
            search_query_results += search_results
            max_id = search_results[-1].GetId()
            #if len(search_query_results) > 15:
            #    break #######
        #print "\n"
        print search_query, ":", len(search_query_results)
        #for s in search_query_results:
        #    print " ", get_status_text(s)
        #print "\n"
        total_search_results += search_query_results
    return total_search_results

if __name__ == "__main__":
    #Initial settings
    os.environ['TZ'] = 'US/Eastern'; time.tzset()

    #look at the results for each query to see if it's returning results expected (its probably good but a little concerned about spaces and &)
    
    collect_tweets("spx", ["S&amp;P 500", "SANDP 500", "SP500", "#SPX", "$SPX"])
    collect_tweets("aapl", ["Apple Inc.", "AAPL", "#AAPL", "$AAPL"])
    collect_tweets("msft", ["Microsoft Corp", "MSFT", "#MSFT", "$MSFT"])
    collect_tweets("xom", ["Exxon", "XOM", "#XOM", "$XOM"])
    collect_tweets("jnj", ["#JNJ", "$JNJ"])
    collect_tweets("ge", ["General Electric", "$GE"])
    collect_tweets("brk", ["Berkshire Hathaway B", "BRK.B", "#BRK.B", "$BRK.B", "$BRK-B"])  #the "b"?
    collect_tweets("fb", ["Facebook Inc", "$FB"])
    collect_tweets("t", ["AT&amp;T Inc", "$T", "$ATT"]) #yikes: T, #T     better?: #ATT
    collect_tweets("amzn", ["Amazon.com Inc", "#AMZN", "$AMZN"])
    collect_tweets("wfc", ["Wells Fargo &amp; Co", "$WFC", "Wells Fargo"])
    
    




    
            