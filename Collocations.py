# Jarvis Miller
# jarvm
from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
from collections import Counter
import string
from itertools import dropwhile
import numpy as np
from io import open
import sys

class Collocation(object):
    punc = set(string.punctuation)

    def __init__(self, file_to_path):
        self.file = file_to_path

    def get_ngrams(self, sentence, N):
        # assuming same word wont be repeated twice o.o
        grams = [tuple(sentence[i:i+N]) for i in xrange(len(sentence)-N+1)]
        # get rid of false bigrams such as ["today", ","]
        # and false unigrams such as ["'"]
        for li in grams[:]:
            if len(self.punc.intersection(li)) > 0:
                grams.remove(li)

        return grams

    def ngram_counter(self, N, min_freq=5):
        ngram_counts = Counter()
        with open(self.file) as corpus:
            for l in corpus:
                ngram_counts.update(Counter((self.get_ngrams(l.split(), N))))

        for key, count in dropwhile(lambda key_count: key_count[1] >= min_freq, ngram_counts.most_common()):
            del ngram_counts[key]
        return ngram_counts

    def PMI(self, ngram_obj):
        # use this for p(word)
        unigram = dict(self.ngram_counter(N=1, min_freq=1))
        pmi_dict = {}

        # index through keys
        for key, val in ngram_obj.items():

            p_w1 = unigram[(key[0],)]
            p_w2 = unigram[(key[1],)]
            pmi_dict[key] = np.log(val / (p_w1 * p_w2))

        return pmi_dict

    def chi_square(self, ngram_obj):
        ngram_dict = dict(ngram_obj)
        tot_occurences = sum(ngram_dict.values())
        chi_sq_dict = {}

        for key, val in ngram_obj.items():
            o_11 = val
            o_22, o_12, o_21 = 0, 0, 0
            w1 = key[0]
            w2 = key[1]
            for key_dict, val_dict in ngram_dict.items():
                # first word same, second diff

                if (w1 != key_dict[0] and w2 != key_dict[1]):
                    o_22 += val_dict
                elif (w1 == key_dict[0] and w2 != key_dict[1]):
                    o_21 += val_dict
                else: # w1 != key_dict[0] and w2 == key_dict[1]:
                    o_12 += val_dict

            num = (o_11 + o_12 + o_21 + o_22) * ((o_11*o_22 - o_21*o_12)**2)
            denom = (o_11 + o_21) * (o_11 + o_12) * (o_12 + o_22) * (o_21 + o_22)
            chi_sq_dict[key] = num / denom

        return chi_sq_dict


if __name__ == "__main__":
    file_path = sys.argv[1]
    measure_param = sys.argv[2]

    coll = Collocation(str(file_path))


    b = coll.ngram_counter(2)
    if measure_param == "PMI":
        print "PMI results:"

        pmi = coll.PMI(b)
        for c in Counter(pmi).most_common(20):
            print c
    else:
        print "Chi square results (this takes a while):"
        chi_sq = coll.chi_square(b)
        for c in Counter(chi_sq).most_common(20):
            print c