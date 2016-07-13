# SP500-Stock-Market-Prediction Algorithm

This project predicts the daily price return of the S&P 500 based on tweets about the index.  It was used for my Master's Project.  Read "MastersProjectReportNoSignatures.pdf" to discover my approach to this problem.



Prerequisites:

Python 2.x, Pandas, Numpy, NLTK, Scikit-Learn. I suggest installing Anaconda ( https://www.continuum.io/downloads ) which has all those libraries in one package.
TensorFlow ( https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html )



Steps to follow to run the project:

1) Clone or download the repository.

2) Download latest S&P 500 closing price data from yahoo or use the csv file in repository.  Go to http://ichart.finance.yahoo.com/table.csv?s=^GSPC&c=2010 , save and rename table.csv to gspc.csv.  

2) Run "get_stock_data.py" which outputs "daily_price_changes.csv" which provides the prediction labels.  It also outputs "moving_average_stock_data.csv" which is the simple moving average indicator data that serves as one of the baseline predictors.

3) Run "get_twitter_data.py" to collect more twitter data.  Note:  You need to add your twitter api credentials on line 9.  Go to dev.twitter.com to sign up for an account.

4) Run "create_continuous_text_file.py" which generates "tweets_text.txt" from the collected tweets.

5) Run "word2vec_optimized_mod_obj.py" to output "seman_oreint_dict.p" the SO scores for every word in the vocabulary.  This file is a modification of tensorflow's word2vec implementation ( https://github.com/tensorflow/ ).  Lines 203-207 changes which lexicon to train on. 

6) Run "gen_dataset_n_predicate_n_eval.py" which generates the dataset, makes a predication, and evaluates results.

7) Optional: Run "get_twitter_data.py" to collect more twitter data.


TODO: Include previous version from my project in Manfred's advanced machine learning class that used used a SGD LR predictor regularized by "sleeping".


