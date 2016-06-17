# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-threaded word2vec unbatched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does true SGD (i.e. no minibatching). To do this efficiently, custom
ops are used to sequentially process data within a 'batch'.

The key ops used are:
* skipgram custom op that does input processing.
* neg_train custom op that efficiently calculates and applies the gradient using
  true SGD.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time
import random
import pickle

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf

from tensorflow.models.embedding import gen_word2vec as word2vec

flags = tf.app.flags

flags.DEFINE_string("save_path", ".", "Directory to write the model.")
flags.DEFINE_string(
    "train_data", "./data/twitter/tweets_text.txt",
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", "./data/eval/questions-words.txt", "Analogy questions. "
    "https://word2vec.googlecode.com/svn/trunk/questions-words.txt.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500,
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")

FLAGS = flags.FLAGS


class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.

    # The training text file.
    self.train_data = FLAGS.train_data

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # Where to write out summaries.
    self.save_path = FLAGS.save_path

    # Eval options.

    # The text file for eval.
    self.eval_data = FLAGS.eval_data


class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self._options = options
    self._session = session
    self._word2id = {}
    self._id2word = []            
    self.build_graph()
    self.build_eval_graph()
    self.save_vocab()
    self._read_analogies()
    
  def _read_analogies(self):
    """Reads through the analogy question file.

    Returns:
      questions: a [n, 4] numpy array containing the analogy question's
                 word ids.
      questions_skipped: questions skipped due to unknown words.
    """
    questions = []
    questions_skipped = 0
    with open(self._options.eval_data, "rb") as analogy_f:
      for line in analogy_f:
        if line.startswith(b":"):  # Skip comments.
          continue
        words = line.strip().lower().split(b" ")
        ids = [self._word2id.get(w.strip()) for w in words]
        if None in ids or len(ids) != 4:
          questions_skipped += 1
        else:
          questions.append(np.array(ids))
    print("Eval analogy file: ", self._options.eval_data)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    self._analogy_questions = np.array(questions, dtype=np.int32)

  def build_graph(self):
    """Build the model graph."""
    opts = self._options

    # The training data. A text file.
    (words, counts, words_per_epoch, current_epoch, total_words_processed,
     examples, labels) = word2vec.skipgram(filename=opts.train_data,
                                           batch_size=opts.batch_size,
                                           window_size=opts.window_size,
                                           min_count=opts.min_count,
                                           subsample=opts.subsample)
    (opts.vocab_words, opts.vocab_counts,
     opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
    opts.vocab_size = len(opts.vocab_words)
    print("Data file: ", opts.train_data)
    print("Vocab size: ", opts.vocab_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)    

    self._id2word = opts.vocab_words
    for i, w in enumerate(self._id2word):
      self._word2id[w] = i
      
    #let me interrupt and get pos/neg words in vocab    
    
    SOCIAL = False
    SOL = True ; num_words = 100
    LOVEHATE = False
    FIN = False
    BULLBEAR = False
    
    if (SOCIAL): #use one of the social polar lexicons
        fileNames = ["./data/train_lexicons/10_social_employment_opportunities.txt", 
                     "./data/train_lexicons/10_social_freedom_from_discrimination.txt", 
                     "./data/train_lexicons/10_social_good_education.txt", 
                     "./data/train_lexicons/10_social_honest_and_responsive_government.txt", 
                     "./data/train_lexicons/10_social_political_freedom.txt"]
        social_issue_to_use = 0
        polarity_dict = pd.read_csv(fileNames[social_issue_to_use], header=None, names=["pos", "neg"])
        neg_terms = polarity_dict["neg"]
        pos_terms = polarity_dict["pos"]
        
    elif(SOL): #use the stock opinion lexicon
        neg_terms = pd.read_csv("./data/train_lexicons/sol_train_neg.csv", header=0)
        pos_terms = pd.read_csv("./data/train_lexicons/sol_train_pos.csv", header=0)
        ordered = True
        if (ordered):
            neg_terms = neg_terms.sort_values(["v1"], axis=0, ascending=True)["w1"].iloc[:num_words]
            pos_terms = pos_terms.sort_values(["v1"], axis=0, ascending=False)["w1"].iloc[:num_words]
        else:
            neg_terms = neg_terms.sample(n=num_words)["w1"]
            pos_terms = pos_terms.sample(n=num_words)["w1"]
            
    elif(LOVEHATE): #use the love-hate lexicon
        justlovehate = True
        neg_terms = pd.read_csv("./data/train_lexicons/hate.txt", names=["neg"])
        pos_terms = pd.read_csv("./data/train_lexicons/love.txt", names=["pos"])
        if(justlovehate):
            neg_terms = neg_terms["neg"].iloc[-1]
            pos_terms = pos_terms["pos"].iloc[-1]
        else:
            neg_terms = neg_terms["neg"].iloc[15:]
            pos_terms = pos_terms["pos"].iloc[15:]
            
    elif(FIN): #use FIN lexicon
        neg_terms = pd.read_csv("./data/train_lexicons/fin_negatives.csv", header=0, index_col=0)["negs"]
        pos_terms = pd.read_csv("./data/train_lexicons/fin_positives.csv", header=0, index_col=0)["poss"]
        
    elif(BULLBEAR):
        neg_terms = ["bearish"]
        pos_terms = ["bullish"]
        
    self.neg_terms_in_vocab = []
    self.neg_ids = []
    self.pos_terms_in_vocab = []
    self.pos_ids = []
    opts = self._options
    for neg_term in neg_terms:
        neg_term = neg_term.encode()
        if neg_term in opts.vocab_words:
            self.neg_terms_in_vocab.append(neg_term)
            self.neg_ids.append(self._word2id.get(neg_term, 0))
    self.neg_ids = tf.constant(self.neg_ids)
    for pos_term in pos_terms:
        pos_term = pos_term.encode()
        if pos_term in opts.vocab_words:
            self.pos_terms_in_vocab.append(pos_term)
            self.pos_ids.append(self._word2id.get(pos_term, 0))
    self.pos_ids = tf.constant(self.pos_ids)
    
    if (LOVEHATE):
        #evaluation only works for the love-hate lexicon ...
        self.eval_neg_id = [self._word2id.get("hate", 0)]
        self.eval_neg_id = tf.constant(self.eval_neg_id)
        self.eval_pos_id = [self._word2id.get("love", 0)]
        self.eval_pos_id = tf.constant(self.eval_pos_id)
        
        #eval neg_ids and pos_ids (all train words)
        neg_terms = pd.read_csv("./data/train_lexicons/hate.txt", names=["neg"])
        pos_terms = pd.read_csv("./data/train_lexicons/love.txt", names=["pos"])
        neg_terms = neg_terms["neg"].iloc[15:]
        pos_terms = pos_terms["pos"].iloc[15:]
        self.eval_neg_ids = []
        self.eval_pos_ids = []
        opts = self._options
        for neg_term in neg_terms:
            neg_term = neg_term.encode()
            if neg_term in opts.vocab_words:
                self.eval_neg_ids.append(self._word2id.get(neg_term, 0))
        self.eval_neg_ids = tf.constant(self.eval_neg_ids)
        for pos_term in pos_terms:
            pos_term = pos_term.encode()
            if pos_term in opts.vocab_words:
                self.eval_pos_ids.append(self._word2id.get(pos_term, 0))
        self.eval_pos_ids = tf.constant(self.eval_pos_ids)
    else:
        neg_terms = pd.read_csv("./data/train_lexicons/sol_train_neg.csv", names=["neg"])
        pos_terms = pd.read_csv("./data/train_lexicons/sol_train_pos.csv", names=["pos"])
        neg_terms = neg_terms["neg"].iloc[num_words:]
        pos_terms = pos_terms["pos"].iloc[num_words:]
        self.eval_neg_ids = []
        self.eval_pos_ids = []
        opts = self._options
        for neg_term in neg_terms:
            neg_term = neg_term.encode()
            if neg_term in opts.vocab_words:
                self.eval_neg_ids.append(self._word2id.get(neg_term, 0))
        self.eval_neg_ids = tf.constant(self.eval_neg_ids)
        for pos_term in pos_terms:
            pos_term = pos_term.encode()
            if pos_term in opts.vocab_words:
                self.eval_pos_ids.append(self._word2id.get(pos_term, 0))
        self.eval_pos_ids = tf.constant(self.eval_pos_ids)
    

    #continue where it left off
    # Declare all variables we need.
    # Input words embedding: [vocab_size, emb_dim]
    w_in = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size,
             opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
        name="w_in")

    # Global step: scalar, i.e., shape [].
    w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="w_out")

    # Global step: []
    global_step = tf.Variable(0, name="global_step")

    # Linear learning rate decay.
    words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
    lr = opts.learning_rate * tf.maximum(
        0.0001,
        1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

    # Training nodes.
    inc = global_step.assign_add(1)
    with tf.control_dependencies([inc]):
      train = word2vec.neg_train(w_in,
                                 w_out,
                                 examples,
                                 labels,
                                 lr,
                                 vocab_count=opts.vocab_counts.tolist(),
                                 num_negative_samples=opts.num_samples)

    self._w_in = w_in
    self._examples = examples
    self._labels = labels
    self._lr = lr
    self._train = train
    self.step = global_step
    self._epoch = current_epoch
    self._words = total_words_processed
    
    #Train nodes with antonyms
    loss2 = self.antonym_loss_and_optimize()
    self._loss2 = loss2
    
  def similarity(self, w1_id, w2_id):
    opts = self._options
    
    #Get embeddings  
    w1_emb = tf.gather(self._w_in, w1_id) 
    w2_emb = tf.gather(self._w_in, w2_id)  
    
    #Compute cosine similarity (ie. loss). TF doesn't have a function for this so I have to 
    #1) get unit norm, 
    w1_emb_norm = tf.nn.l2_normalize(w1_emb,0)
    w2_emb_norm = tf.nn.l2_normalize(w2_emb,0)
    #2) Change to 2D array 
    w1_emb_norm_2d = tf.reshape(w1_emb_norm, [1,opts.emb_dim])
    w2_emb_norm_2d = tf.reshape(w2_emb_norm, [opts.emb_dim,1])
    #3) matrix multiple 
    dist = tf.matmul(w1_emb_norm_2d, w2_emb_norm_2d)
    #4) Change to 1D array
    dist = tf.reshape(dist, [])
    
    return dist

  def antonym_loss_and_optimize(self):
    opts = self._options
    
    shuffled_neg_ids = tf.random_shuffle(self.neg_ids)
    shuffled_pos_ids = tf.random_shuffle(self.pos_ids)
    
    #calc similarities of a sampled neg_terms synonym
    w1_id = shuffled_neg_ids[0]
    w2_id = shuffled_neg_ids[1]
    objective = self.similarity(w1_id, w2_id)
    
    #calc similarities of a sampled pos_terms synonym
    w1_id = shuffled_pos_ids[0]
    w2_id = shuffled_pos_ids[1]
    objective = tf.add(self.similarity(w1_id, w2_id), objective)
    
    #calc similarities of a sampled antonym
    w1_id = shuffled_neg_ids[0]
    w2_id = shuffled_pos_ids[0]
    objective = tf.add(tf.neg(self.similarity(w1_id, w2_id)), objective)
    
    #calc similarities of a sampled antonym again
    w1_id = shuffled_neg_ids[1]
    w2_id = shuffled_pos_ids[1]
    objective = tf.add(tf.neg(self.similarity(w1_id, w2_id)), objective)
    
    #change objective to loss
    antonym_loss = tf.neg(objective)
    
    #Perform optimization
    optimizer = tf.train.GradientDescentOptimizer(self._lr) #0.0001) #self._lr)
    train2 = optimizer.minimize(antonym_loss)
    self._train2 = train2
    
    return antonym_loss
    
  def save_vocab(self):
    """Save the vocabulary to a file so the model can be reloaded."""
    opts = self._options
    with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
      for i in xrange(opts.vocab_size):
        f.write("%s %d\n" % (tf.compat.as_text(opts.vocab_words[i]),
                             opts.vocab_counts[i]))

  def build_eval_graph(self):
    """Build the evaluation graph."""
    # Eval graph
    opts = self._options

    # Each analogy task is to predict the 4th word (d) given three
    # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
    # predict d=paris.

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(self._w_in, 1)

    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    a_emb = tf.gather(nemb, analogy_a)  # a's embs
    b_emb = tf.gather(nemb, analogy_b)  # b's embs
    c_emb = tf.gather(nemb, analogy_c)  # c's embs

    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    target = c_emb + (b_emb - a_emb)

    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    dist = tf.matmul(target, nemb, transpose_b=True)

    # For each question (row in dist), find the top 4 words.
    _, pred_idx = tf.nn.top_k(dist, 4)

    # Nodes for computing neighbors for a given word according to
    # their cosine distance.
    nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    nearby_emb = tf.gather(nemb, nearby_word)
    nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                         min(1000, opts.vocab_size))

    # Nodes in the construct graph which are used by training and
    # evaluation to run/feed/fetch.
    self._analogy_a = analogy_a
    self._analogy_b = analogy_b
    self._analogy_c = analogy_c
    self._analogy_pred_idx = pred_idx
    self._nearby_word = nearby_word
    self._nearby_val = nearby_val
    self._nearby_idx = nearby_idx

    # Properly initialize all variables.
    tf.initialize_all_variables().run()

    self.saver = tf.train.Saver()

  def _train_thread_body(self):
    initial_epoch, = self._session.run([self._epoch])
    count = 0
    while True:
      antonym_loss = True
      if (antonym_loss):
          count += 1
          if count == 1:
              _, _, epoch = self._session.run([self._train, self._train2, self._epoch])
              count = 0
          else:
              _, epoch = self._session.run([self._train, self._epoch])
      else:
          _, epoch = self._session.run([self._train, self._epoch])
      
      if epoch != initial_epoch:
        break

  def train(self):
    """Train the model."""
    opts = self._options

    initial_epoch, initial_words = self._session.run([self._epoch, self._words])

    workers = []
    for _ in xrange(opts.concurrent_steps):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)

    last_words, last_time = initial_words, time.time()
    while True:
      time.sleep(10)  # Reports our progress once a while.
      
      antonym_loss = True
      if (antonym_loss):
          (epoch, step, loss2, words, lr) = self._session.run(
              [self._epoch, self.step, self._loss2, self._words, self._lr])
      else:
          (epoch, step, words, lr) = self._session.run(
              [self._epoch, self.step, self._words, self._lr])       
       
      now = time.time()
      last_words, last_time, rate = words, now, (words - last_words) / (
          now - last_time)
      if (antonym_loss):
          print("Epoch %d Step %8d: lr = %.3f loss2 = %.3f  words/sec = %.0f\r" %
              (epoch, step, lr, loss2, rate), end="")
      else:
          print("Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r" %
              (epoch, step, lr, rate), end="")
      
      
      sys.stdout.flush()
      if epoch != initial_epoch:
        break

    for t in workers:
      t.join()

  def _predict(self, analogy):
    """Predict the top 4 answers for analogy questions."""
    idx, = self._session.run([self._analogy_pred_idx], {
        self._analogy_a: analogy[:, 0],
        self._analogy_b: analogy[:, 1],
        self._analogy_c: analogy[:, 2]
    })
    return idx

  def eval(self):
    """Evaluate analogy questions and reports accuracy."""

    # How many questions we get right at precision@1.
    correct = 0

    total = self._analogy_questions.shape[0]
    start = 0
    while start < total:
      limit = start + 2500
      sub = self._analogy_questions[start:limit, :]
      idx = self._predict(sub)
      start = limit
      for question in xrange(sub.shape[0]):
        for j in xrange(4):
          if idx[question, j] == sub[question, 3]:
            # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
            correct += 1
            break
          elif idx[question, j] in sub[question, :3]:
            # We need to skip words already in the question.
            continue
          else:
            # The correct label is not the precision@1
            break
    print()
    print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                              correct * 100.0 / total))

  def analogy(self, w0, w1, w2):
    """Predict word w3 as in w0:w1 vs w2:w3."""
    wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
    idx = self._predict(wid)
    for c in [self._id2word[i] for i in idx[0, :]]:
      if c not in [w0, w1, w2]:
        return c
    return "unknown"

  def nearby(self, words, num=20):
    """Prints out nearby words given a list of words."""
    ids = np.array([self._word2id.get(x, 0) for x in words])
    vals, idx = self._session.run(
        [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
    for i in xrange(len(words)):
      print("\n%s\n=====================================" % (words[i]))
      for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
        print("%-20s %6.4f" % (self._id2word[neighbor], distance))
        
  def eval2(self, name, eval_pos_ids, eval_neg_ids):
    opts = self._options
    
    so_dict = self.most_similar(eval_pos_ids, eval_neg_ids)
    #save my most similar
    pickle.dump(so_dict, open("seman_oreint_dict.p", "w+"))
    
    correct = 0 #calc accuracy: sum of true positives + sum false positives / total population
    total_population = 0  

    avg_neg_score = 0
    avg_pos_score = 0
    count = 0

    neg_terms = pd.read_csv("hate.txt", names=["neg"])
    neg_terms = neg_terms["neg"].iloc[:15]
    for neg_term in neg_terms:
        neg_term = neg_term.encode()
        if neg_term in opts.vocab_words:
            print(so_dict[neg_term])
            if so_dict[neg_term] < 0:
                correct += 1
            avg_neg_score += so_dict[neg_term] 
            count += 1
            total_population += 1
    
    if count != 0:
        avg_neg_score = float(avg_neg_score)/count
    count = 0
    
    print("\n\n")
            
    pos_terms = pd.read_csv("love.txt", names=["pos"])
    pos_terms = pos_terms["pos"].iloc[:15]
    for pos_term in pos_terms:
        pos_term = pos_term.encode()
        if pos_term in opts.vocab_words:
            print(so_dict[pos_term])
            if so_dict[pos_term] > 0:
                correct += 1
            avg_pos_score += so_dict[pos_term] 
            count += 1
            total_population += 1
    
    if count != 0:
        avg_pos_score = float(avg_pos_score)/count

    print()
    print("Eval%s %4d/%d accuracy = %4.1f%%, avg_pos_score = %0.2f, avg_neg_score = %0.2f" % (name, correct, total_population,
                                              correct * 100.0 / total_population, avg_pos_score, avg_neg_score))
    
  def most_similar(self, eval_pos_ids, eval_neg_ids):
    opts = self._options
    #using pos/negs, return a key-value-dictionary where the keys are words in the 
    #vocabulary & values is the semantic oreintation relative to pos/neg words
    so_dict = {}    
    
    _w_in = self._session.run(self._w_in)
    pos_ids, neg_ids = self._session.run([eval_pos_ids, eval_neg_ids])
    
    for w1 in opts.vocab_words:
      w1_id = self._word2id.get(w1, 0)
      similarity_matrix = cosine_similarity(_w_in[np.array(pos_ids)], _w_in[w1_id].reshape(1,-1))
      #print("pos sim matrix:", similarity_matrix)
      
      # for all pos/neg words get similarities to w1 in order to calc sum of pos sims/pos word count - sum of neg sims/neg word count 
      pos_sim = 0
      for w2_id in range(len(pos_ids)):
          #pos_sim = similarity_matrix[w1_id,w2_id] + pos_sim
          pos_sim = similarity_matrix[w2_id] + pos_sim
      pos_sim = float(pos_sim)/len(pos_ids)
      #print("average pos_sim", pos_sim)
      
      similarity_matrix = cosine_similarity(_w_in[np.array(neg_ids)], _w_in[w1_id].reshape(1,-1))
      neg_sim = 0
      for w2_id in range(len(neg_ids)):
          neg_sim = similarity_matrix[w2_id] + neg_sim
      neg_sim = float(neg_sim)/len(neg_ids)
      #print("average neg_sim", neg_sim)
      
      so_dict[w1] = pos_sim - neg_sim
      #print("so_dict[", w1, "]:", so_dict[w1])

    return so_dict



def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


def main(_):
  """Train a word2vec model."""
  if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
    print("--train_data --eval_data and --save_path must be specified.")
    sys.exit(1)
  opts = Options()
  with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      model = Word2Vec(opts, session)
    for _ in xrange(opts.epochs_to_train):
      model.train()  # Process one epoch
      #print("\n\n")
      model.eval()  # Eval analogies.
      model.eval2("2", model.eval_pos_ids, model.eval_neg_ids) # Eval every test pos/neg words using all train words in SO equation
      #model.eval2("3", model.eval_pos_id, model.eval_neg_id) # Used to make initial love-hate histograms. Eval every test pos/neg words using one neg/one pos word in SO equation 
      
    # Perform a final save.
    model.saver.save(session, os.path.join(opts.save_path, "model.ckpt"),
                     global_step=model.step)
    if FLAGS.interactive:
      # E.g.,
      # [0]: model.analogy(b'france', b'paris', b'russia')
      # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
      _start_shell(locals())


if __name__ == "__main__":
  tf.app.run()
