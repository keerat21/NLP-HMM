import numpy as np #hint: np.log
from collections import defaultdict,Counter
from gtnlplib import scorer, most_common,preproc
from gtnlplib.constants import OFFSET

def learnNBWeights(counts, class_counts, allkeys, alpha=0.1):
    '''
    : param counts :        word counts for every label in the training data
    : param class_counts :  the count of instances with every label in the training data
    : param allkeys :       list of all word types that are observed in the training data

    Chose algorithm 2 from Eisenstein NLP textbook. From class_counts the prior and the log 
    likelyhood are drawn categorically, because the instances are identically distributed. 
    Log prior is calculated using the counts from a class conditioned on the total documents 
    in the training set. 

    : returns :             weights
    : rtype :               defaultdict
    '''
    weights = defaultdict(int)
    doc_count = sum(class_counts.values())
    smoothing = (len(allkeys) - 1) * alpha

    for label, count in class_counts.items():
        # log mu for the offset, which parametrizes the prior log P(y), relative freq
        weights[(label, OFFSET)] = np.log(float(count) / doc_count)
        # log phi for the word counts, which parametrizes the likelihood log P(x | y)
        # normalized by the class counts of the current label
        for feature in allkeys:
            weights[(label, feature)] = np.log((counts[label][feature] + alpha)
                                               / (class_counts[label] + smoothing))
    return defaultdict(int, weights)
