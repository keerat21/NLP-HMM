import numpy as np #hint: np.log
from itertools import chain
import operator
from collections import defaultdict, Counter
from gtnlplib.preproc import conllSeqGenerator

from gtnlplib import scorer
from gtnlplib import most_common
from gtnlplib import preproc
from gtnlplib.constants import START_TAG, TRANS, END_TAG, EMIT
from gtnlplib import naivebayes

def argmax(scores):
    """Find the key that has the highest value in the scores dict"""
    return max(scores.iteritems(),key=operator.itemgetter(1))[0]

# define viterbiTagger
def viterbiTagger(words,feat_func,weights,all_tags,debug=False):
    """Tag the given words using the viterbi algorithm
        Parameters:
        words -- A list of tokens to tag
        feat_func -- A function of (words, curr_tag, prev_tag, curr_index)
        that produces features
        weights -- A defaultdict that maps features to numeric score. Should
        not key error for indexing into keys that do not exist.
        all_tags -- A set of all possible tags
        debug -- (optional) If True, print the trellis at each layer
        Returns:
        tuple of (tags, best_score), where
        tags -- The highest scoring sequence of tags (list of tags s.t. tags[i]
        is the tag of words[i])
        best_score -- The highest score of any sequence of tags
        """
    trellis = [None] * len(words)
    pointers = [None] * len(words)
    output = [None] * len(words)
    for i in range(len(words)):         #traverse the length of words
        pointer = defaultdict(str) 
        scores = defaultdict(float) 
        for curr_tag in all_tags:   #get current tag
            if i == 0:
                prev_tag = START_TAG #prv tag is start_tag at i==0
                pointer[curr_tag] = prev_tag 
                feat = feat_func(words,curr_tag,prev_tag,i) #feat[0] is emit probability and feat[1] is transition prob
                scores[curr_tag] = weights[feat[0]] + weights[feat[1]] #add probs of transition and emmision prob
            else:
                max_score= -np.inf
                ptag = None
                for prev_tag in all_tags: #if not start there was previous tag
                    feat = feat_func(words,curr_tag,prev_tag,i) #get transmission and emmision features using previous and current tags
                    if max_score < trellis[i-1][prev_tag] + weights[feat[0]] + weights[feat[1]]: #trellis contains the values of previous features
                        max_score = trellis[i-1][prev_tag] + weights[feat[0]] + weights[feat[1]] #add probs of trellis, transition and emmision prob to score
                        ptag = prev_tag #point to previous tag`
                pointer[curr_tag] = ptag
                scores[curr_tag] = max_score
        trellis[i] = scores
        pointers[i] = pointer

    #deal with the last word
    curr_tag = END_TAG
    max_score = -np.inf
    ptr_last = None
    for prev_tag in all_tags:
        feat = feat_func(words,curr_tag,prev_tag, len(words))
        if max_score < trellis[len(words)-1][prev_tag] + weights[feat[0]]:  #for last tag we will have one feat that is transition
            max_score = trellis[len(words)-1][prev_tag] + weights[feat[0]]
            ptr_last = prev_tag 

    best_score = max_score
    output = [ptr_last]
    prev_tag = ptr_last
    for i in range(len(words))[::-1]:
        output.append(pointers[i][prev_tag])
        prev_tag = pointers[i][prev_tag]
    output = output[::-1][1:]
    return output,best_score

def get_HMM_weights(trainfile):
    """Train a set of of log-prob weights using HMM transition model
        Parameters:
        trainfile -- The name of the file to train weights
        Returns:
        weights -- Weights dict with log-prob of transition and emit features
        """
    # compute naive bayes weights
    hmm_weights = defaultdict(lambda : -1000.)
    
    counters = most_common.get_tags(trainfile)     #Counter of occurences of word in each tag
    class_counts = most_common.get_class_counts(counters) #Counts of total word occurences in each tag
    allwords = set()
    for counts in counters.values(): #counts is word with its num of occurances
        allwords.update(set(counts.keys())) #allwords contains all the words from the counter
    nb_weights = naivebayes.learnNBWeights(counters,class_counts,allwords)  #get weights for each 
    for key, val in nb_weights.iteritems():
        hmm_weights[key[0],key[1],EMIT] = val       #hmm_weights is the weight from naive bayes

    #get transmission weights
    trans_counter = Counter()
    num_insts = 0
    for i, (words, tags) in enumerate(preproc.conllSeqGenerator(trainfile)):
        num_insts += 1              #number of statements
        for j in range(len(tags)):  #traverse ecah tag in statement
            if j == 0:      #first tag will be Start_tag
                trans_counter[(tags[j],START_TAG)] += 1
            elif j == len(tags) - 1:
                trans_counter[(END_TAG,tags[j])] += 1  #last tag will be end_tag
            else:
                trans_counter[(tags[j+1],tags[j])] += 1 #add value for a tag transition pair
    trans_weights = defaultdict(float)    
    for tag, count in class_counts.iteritems(): #each tag and its total words in that tag
        for t_tag in class_counts.keys():         #traverse each tag in class_counts
            trans_weights[(t_tag, tag)] = np.log(float(trans_counter[(t_tag, tag)])/count) #num of t_tag after tag divided by total words in that tag
        trans_weights[(tag,START_TAG) ] = np.log(float(trans_counter[(tag, START_TAG) ])/num_insts) #prob of each starting tag
        trans_weights[(END_TAG,tag)] = np.log(float(trans_counter[(END_TAG,tag)])/num_insts) #prob of each ending tag
    for key, val in trans_weights.iteritems():
        hmm_weights[key[0],key[1],TRANS] = val #prob of each transition from current to next tag

    return hmm_weights



def hmm_feats(words,curr_tag,prev_tag,i):
    """Feature function for HMM that returns emit and transition features"""
    if i < len(words):
        return [(curr_tag,words[i],EMIT),(curr_tag,prev_tag,TRANS)]
    else:
        return [(curr_tag,prev_tag,TRANS)]