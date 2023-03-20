''' your code '''
import operator
from  constants import *
from collections import defaultdict, Counter
from gtnlplib import preproc
import scorer
from gtnlplib import constants
from gtnlplib import clf_base

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

def get_tags(trainfile):
    """Produce a Counter of occurences of word in each tag"""
    counters = defaultdict(Counter)             #Have to aggregate the  index(tags); each tag points to the list of words
    for words,tags in preproc.conllSeqGenerator(trainfile):     #lists of each sentence words and tags
        for j in range(len(words)):     #each word, tag from the list of of words and tags
            counters[tags[j]][words[j]] += 1    #tags will be aggregated, for each tag, list of words and their count
        
    return counters

def get_noun_weights():
    """Produce weights dict mapping all words as noun"""
    your_weights = defaultdict(int)
    your_weights[('N', constants.OFFSET)] = 1 #Noun weight is 1
    
    return your_weights

def get_most_common_weights(trainfile):
    
    """ get weights for each tag, word pair
        Parameters:
        trainfile -- has words and tags
        Returns:
        weights -- weights for each word, tag pair and each tag
        """
    counters = get_tags(trainfile)  #Counter of occurences of word in each tag
    
    counts = get_class_counts(counters) #Counts of total word occurences in each tag
    weights = defaultdict(int)
    
    
    for tag,tag_ctr in counters.iteritems():
        weights[tag,constants.OFFSET] = counts[tag]     #weights of each tag (total counts)
        for word in tag_ctr:
            weights[tag,word] = tag_ctr[word] #weights of each tag, word pair(total tag,word pair counts)
  
 

    return weights

def get_class_counts(counters):
#"""get total count of word occurances in each tag
#        Parameters:
#        counters -- #Counter of occurences of word in each tag
#        Returns:
#        counts -- total word counts in each tag
#"""    
    counts = defaultdict(int)
    
    for tag,tag_ctr in counters.iteritems(): #go through each tag, and and list of word occurances in it
        for word in tag_ctr:    #go in each word occurance in the specific tag
            counts[tag] += tag_ctr[word]  #add all the word occurances
    
    return counts
