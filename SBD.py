# Jarvis Miller
# jarvm

from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
import pandas as pd
import numpy as np
import csv
from sklearn import tree
import os
from io import open
from sklearn.metrics import confusion_matrix
import sys



df_train = pd.read_table('SBD.train', sep=' ', header=None, quoting=csv.QUOTE_NONE,
                 engine='python', usecols=xrange(1,3))
df_test = pd.read_csv('SBD.test', sep=' ', header=None, quoting=csv.QUOTE_NONE,
                 engine='python', usecols=xrange(1,3))



def create_features(df, full_features=True):
# extract 5 core features
    df.columns = ['word', 'label']
    df = pd.concat([df,pd.DataFrame(columns=["word", "label", "l_word", "r_word", "l_word_lt_3",
                                             "l_cap", "r_cap", "r_word_start_t"])])
#     df["r_word"] = df["word"].apply(lambda x: str(x).partition(".")[2] if str(x).partition(".")[2] != '' else df.word.shift(-1).str.strip())
    df["l_word"] = df["word"].apply(lambda x: str(x).partition(".")[0])
    # When it's just a 'period', just go to the previous row
    df.loc[df.word == ".", 'l_word'] = df.loc[df.word == ".", 'word'].shift(1).str.strip()

    df['r_word'] = df.word.shift(-1).str.strip()
#     df['l_word'] = df.word.shift(1).str.strip()
    df = df[df['label'].str.contains("EOS", "NEOS")]

    df["l_word_lt_3"] = df['l_word'].apply(lambda x: 1 if len(str(x)) < 3 else 0)
    df["l_cap"] = df['l_word'].apply(lambda x: 1 if str(x)[0].isupper() else 0)
    df["r_cap"] = df['r_word'].apply(lambda x: 1 if str(x)[0].isupper() else 0)

    if full_features:
        prefix_list = ["mr", 'mrs', 'ms', 'miss', 'dr', ]
        df["left_word_prefix"] = df['l_word'].apply(lambda x: 1 if str(x).lower() in prefix_list else 0)
        df["r_word_start_t"] = df["r_word"].apply(lambda x: 1 if str(x)[0].lower() == 't' else 0)
        df["r_word_lt_4"] = df["r_word"].apply(lambda x: 1 if len(str(x)) < 4 else 0)

    X = df.ix[:, df.columns != "label"]
    Y = df.label

    return X, Y

def create_dummies(x_train, x_test):
    dummy_train = pd.concat([x_train, pd.get_dummies(x_train.l_word, prefix="l_"),
                         pd.get_dummies(x_train.r_word, prefix="r_")], axis=1)

    dummy_test = pd.concat([x_test, pd.get_dummies(x_test.l_word, prefix="l_"),
                         pd.get_dummies(x_test.r_word, prefix="r_")], axis=1)
    # re-index the new data to the columns of the training data, filling the missing values with 0
    dummy_test = dummy_test.reindex(columns = dummy_train.columns, fill_value=0)

    # get rid of left word and right word
    dummy_test.drop(['l_word', 'r_word', 'word'], inplace=True, axis=1)
    dummy_train.drop(['l_word', 'r_word', 'word'], inplace=True, axis=1)



    return dummy_train, dummy_test


def build_evaluate_tree(X_train, Y_train, X_test, Y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)

    pred = clf.predict(X_test)
    accuracy = np.mean(pred == Y_test)

    return pred, accuracy




if __name__ == "__main__":

    train = sys.argv[1]
    test = sys.argv[2]

    df_train = pd.read_table(str(train), sep=' ', header=None, quoting=csv.QUOTE_NONE,
                     engine='python', usecols=range(1,3))
    df_test = pd.read_csv(str(test), sep=' ', header=None, quoting=csv.QUOTE_NONE,
                     engine='python', usecols=range(1,3))

    x_train, y_train = create_features(df_train)
    x_test, y_test = create_features(df_test)
    x_train, x_test = create_dummies(x_train, x_test)


    pred, accuracy = build_evaluate_tree(x_train, y_train,
                                         x_test, y_test)
    print 'accuracy is: ' + str(accuracy)
    print "confusion matrix: \n" + str(confusion_matrix(y_test, pred, labels=["EOS", 'NEOS']))

    SBD_out = pd.DataFrame([list(df_test.word[df_test['label'].str.contains("EOS", "NEOS")]),
                  list(y_test),
                  list(pred)]).T
    SBD_out.columns = ["word", "true_label", "pred_label"]

    SBD_out.to_csv("SBD.out", index=False, sep = ' ')



