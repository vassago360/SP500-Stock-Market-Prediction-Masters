import os, time, csv
import pandas as pd
import numpy as np

def create_daily_price_changes_csv():
    daily_price_changes = pd.DataFrame([], columns=["yday", "change in price"]) #yday is the ith day of the year
    gspc = pd.read_csv("gspc.csv")
    gspc = gspc.iloc[::-1]  #reverse rows so that it starts with the earlist date first
    
    for i in gspc.index:
        date = gspc.loc[i]["Date"]
        year = time.strptime(date, "%Y-%m-%d").tm_year
        if year == 2016: # only concerned with getting data from 2016 
            yday = time.strptime(date, "%Y-%m-%d").tm_yday
            if i == 0:
                daily_price_changes.loc[i] = [yday, 0]
            else:
                next_day_price_close = gspc.loc[i-1]["Close"]
                curr_day_price_close = gspc.loc[i]["Close"]
                daily_price_changes.loc[i] = [yday, next_day_price_close - curr_day_price_close] # future close price - current close price
        
    daily_price_changes.to_csv("daily_price_changes.csv", index=False) 

def create_moving_averages_csv(moving_averages):
    gspc = pd.read_csv("gspc.csv")
    gspc = gspc.iloc[::-1]  #reverse rows so that it starts with the earlist date first
    
    # old code i wrote.  i wrote it before i discovered pandas ...
    data = []
    for moving_average in moving_averages:
        stock_prices = []
        stock_prices_diff = []
        dates = []
        for rownum, i in enumerate(gspc.index):
            if rownum == 0: #oldest date
                price_diff_prev_day = 0
                current_close_price = float(gspc.loc[i]["Close"])
            else:
                date = gspc.loc[i]["Date"]
                dates.append(date)
                previous_close_price = current_close_price
                current_close_price = float(gspc.loc[i]["Close"])
                price_diff_prev_day = current_close_price - previous_close_price
                stock_prices.append(current_close_price)
                stock_prices_diff.append(price_diff_prev_day)
        moving_average_data = np.concatenate((np.zeros((moving_average)), 
                                              np.convolve(stock_prices, np.ones((moving_average,))/moving_average, mode='valid')))[:-1]
        moving_average_daily_difference_prediction = moving_average_data - np.asarray([0] + stock_prices)[:-1]
        data.append(moving_average_daily_difference_prediction)   
    data = zip(*([dates] + data))
    
    with open('moving_average_stock_data.csv', 'wb') as fp:
        a = csv.writer(fp, delimiter=',')
        header = ["Date"] + [str(moving_average) + " moving average" for moving_average in moving_averages]
        a.writerows([header]+data)
        
    # need only 2016 data and yday column added. so without cleaning up the code I will reopen the file and make modifications
    moving_average_stock_data = pd.read_csv("moving_average_stock_data.csv")
    new_ma_data = pd.DataFrame([], columns=["yday"]+[str(moving_average) + " moving average" for moving_average in moving_averages])
    for i in moving_average_stock_data.index:
        date = moving_average_stock_data.loc[i]["Date"]
        year = time.strptime(date, "%Y-%m-%d").tm_year
        if year == 2016: # only concerned with getting data from 2016 
            yday = time.strptime(date, "%Y-%m-%d").tm_yday
            new_ma_data.loc[i] = moving_average_stock_data.loc[i]
            #new_ma_data.loc[i] = [yday, moving_average_stock_data.loc[i+1][1:]]
            #pd.concat(pd.Series([4]),moving_average_stock_data.loc[i+1][1:])
    new_ma_data.to_csv("moving_average_stock_data.csv")
    

if __name__ == "__main__":
    os.environ['TZ'] = 'US/Eastern'
    time.tzset()
    
    create_daily_price_changes_csv() # compute and save daily price changes.  It is used as the prediction label 
    create_moving_averages_csv([5, 10, 50])

    
    
    