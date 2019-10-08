#!/usr/bin/env python
#encoding=utf-8

from collections import Counter
from itertools import izip, islice, tee, chain

def longest_common_subsequence(sl, sr): 
    m = [[0 for i in range(len(sr) + 1)] for j in range(len(sl) + 1)] 
    mmax = 0 
    for i in range(1, len(sl) + 1):
        for j in range(1, len(sr) + 1):
            if sl[i-1] == sr[j-1]: 
                m[i][j] = m[i-1][j-1] + 1
            else:
                m[i][j] = max(m[i-1][j], m[i][j-1])
            mmax = max(mmax, m[i][j])

    return mmax

def longest_common_substring(sl, sr):
    m = [[0 for i in range(len(sr) + 1)] for j in range(len(sl) + 1)] 
    mmax = 0 
    for i in range(1, len(sl) + 1):
        for j in range(1, len(sr) + 1):
            if sl[i-1] == sr[j-1]: 
                m[i][j] = m[i-1][j-1] + 1
                mmax = max(mmax, m[i][j])

    return mmax

def longest_common_prefix(sl, sr):
    mmax = 0 
    for cl, cr in zip(sl, sr):
        if cl == cr: 
            mmax += 1
        else:
            break
    return mmax

def longest_common_suffix(sl, sr):
    mmax = 0 
    for cl, cr in zip(reversed(sl), reversed(sr)):
        if cl == cr: 
            mmax += 1
        else:
            break
    return mmax

def levenshtein_distance(sl, sr):
    m = [[0 for i in range(len(sr) + 1)] for j in range(len(sl) + 1)] 
    len_sl = len(sl)
    len_sr = len(sr)
    for i in range(len_sr + 1): m[0][i] = i
    for i in range(len_sl + 1): m[i][0] = i

    for i in range(1, len_sl + 1):
        for j in range(1, len_sr + 1):
            temp = 0
            if sl[i-1] == sr[j-1]: 
                m[i][j] = m[i-1][j-1]
            else:
                m[i][j] = min(m[i-1][j], m[i][j-1], m[i-1][j-1]) + 1

    return m[len_sl][len_sr]

def damerau_levenshtein_distance(sl, sr):
    d = {}
    len_sl = len(sl)
    len_sr = len(sr)
    for i in range(-1, len_sl):
        d[(i,-1)] = i+1
    for j in range(-1, len_sr):
        d[(-1,j)] = j+1

    for i in range(len_sl):
        for j in range(len_sr):
            if sl[i] == sr[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and sl[i]==sr[j-1] and sl[i-1] == sr[j]:
                d[(i,j)] = min(d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[len_sl-1,len_sr-1]

def ngrams(sequence, n, padding=None):
    if padding:
        sequence = chain(['<s>']*(n-1), sequence, ['</s>']*(n-1))

    return izip(*(islice(seq, index, None) for index, seq in enumerate(tee(sequence, n))))

def jaccard_coefficient(sl, sr):
    set_sl = set(sl)
    set_sr = set(sr)
    if not set_sl and not set_sr:
        return 0
    return 1.0 * len(set_sl&set_sr) / len(set_sl|set_sr)

def dice_coefficient(sl, sr):
    set_sl = set(sl)
    set_sr = set(sr)
    if not set_sl and not set_sr:
        return 0
    return 2.0 * len(set_sl&set_sr) / (len(set_sl) + len(set_sr))

def overlap_coefficient(sl, sr):
    set_sl = set(sl)
    set_sr = set(sr)
    if not set_sl or not set_sr:
        return 0
    return  1.0 * len(set_sl&set_sr) / min(len(set_sl), len(set_sr))

def test():

    print list(ngrams('abcd', 3))
    #print list(ngrams('abcd', 3, padding=True))
'''
    print longest_common_subsequence('abced', 'abe')
    print longest_common_substring('abced', 'abe')
    
    print longest_common_prefix('abced', 'abe')
    print longest_common_suffix('abcede', 'abe')

    print damerau_levenshtein_distance('abcede', 'abecde')
    print levenshtein_distance('abcede', 'abecde')
'''
if __name__ == "__main__":
    test()
    
