import math
import random
import re
import string
from collections import Counter

import numpy as np
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# TODO: as part of our dictionary building we should build an index into words that should be treated as one
# some bigrams are New York
# some trigrams are Analytic Hierarchy Process
# some bigrams are Multi criteria decision making
# or we treat the acronyms - irrespective of the acronyms such a word collapser is useful until the n-gram stage

# STEP 1: Learn a vocabulary
# In this step, we're just identifying all the unique words in the training set and counting the number of times they occur.
# We also add to the  vocabulary every combination of two words observed in the text.
# For example, if we have a sentence "I love pizza", then we add vocabulary entries and counts for "I", "love", "pizza", "I_love", and "love_pizza".


# STEP 2: Decide which word combinations represent phrases.
# In this step, we go back through the training text again, and evaluate whether each word combination should be turned into a phrase.
# We are trying to determine whether words A and B should be turned into  A_B.
# The variable 'pa' is the word count for word A, and 'pb' is the count for word B. 'pab' is the word count for A_B.
# Consider the following ratio:    pab / (pa * pb)
#
# This ratio must be a fraction, because pab <= pa and pab <= pb. The fraction will be larger if:
#   - pab is large relative to pa and pb, meaning that when A and B occur they are likely to occur together.
#   - pa and/or pb are small, meaning that words A and B are relatively infrequent.
#
# They modify this ratio slightly by subtracting the "min_count" parameter from pab. This will eliminate very infrequent phrases. The new ratio is (pab - min_count) / (pa * pb)
#
# Finally, this ratio is multiplied by the total number of words in the training text.
# Presumably, this has the effect of making the threshold value more independent of the training set size.

# if the steps one and two are run multiple times, we can move from bigrams to trigrams ...

def onehot_from_dictionary(dictionary, token):
    """
    represent the token as a one-hot vector
    :param dictionary:
    :param token:
    :return:
    """
    assert(isinstance(dictionary, dict))
    assert(isinstance(token, str))

    if token not in dictionary:
        raise ValueError('onehot_from_dictionary - token not defined in dictionary')

    l = dictionary.__len__()
    idx_token = dictionary[token]

    v_onehot = np.zeros(l)
    v_onehot[idx_token] = 1

    return v_onehot


def corpus_to_onehot(dictionary, corpus):
    """
    represent the tokens in the corpus as a one-hot resulting in a one-hot matrix of dimensionality [dict, corpus].
    :param dictionary:
    :param corpus:
    :return:
    """
    assert(isinstance(dictionary, dict))
    assert(isinstance(corpus, str))
    corpus_size = corpus.__len__()
    onehot_size = dictionary.__len__()

    m_onehot = np.zeros(shape=(corpus_size, onehot_size), dtype=np.uint32)

    for i, c in enumerate(corpus):
        idx_c = dictionary[c]
        m_onehot[i, idx_c] = 1

    return m_onehot


def onehot_to_corpus(dictionary, m_onehot):
    """
    Convert a matrix of one-hots back to a text representation
    :param dictionary:
    :param m_onehot:
    :return:
    """
    assert(isinstance(dictionary, dict))

    if m_onehot.shape[1] != dictionary.__len__():
        raise ValueError('onehot_to_corpus - onehot dimensions in matrix and dictionary do not match')

    corpus = ""
    dict_tokens = dictionary.keys()

    for onehot_word in m_onehot:
        idx_t = np.argmax(onehot_word)
        corpus = corpus + dict_tokens[idx_t]

    assert(corpus.__len__() == m_onehot.shape[0])

    return corpus


def fn_preprocess_corpus(replace_dict, verbose = True, save_processed = None):
    """
    :param replace_dict:
    :param verbose:
    :param save_processed:
    :return:
    """
    if replace_dict is not None:
        assert(isinstance(replace_dict, dict))

    if verbose:
        dict_len = 0
        if replace_dict is not None:
            dict_len = replace_dict.__len__()
        print("Replacement dictionary has %s entries ..." % dict_len)

    def preprocess(document):
        # within a document we replace as long as we find a match to replace
        if replace_dict is not None and replace_dict.__len__() > 0:
            for common_token in replace_dict:
                v_replace = replace_dict[common_token]
                for a_to_replace in v_replace:
                    if re.search(a_to_replace, document):
                        document = re.sub(a_to_replace, common_token, document)

        if save_processed is not None:
            with open(save_processed, "a") as myfile:
                myfile.write('\n<Document>\n')
                myfile.write(document)
                myfile.write('\n</Document>')

        return document

    return preprocess


def fn_tokenize_word(ngram=2,
                     remove_stopwords=True,
                     remove_numbers=True,
                     remove_punctuation=True):
    """
    higher order function that creates the actual tokenization function, which can be called within the python pipeline
    :param ngram: int=the type of ngram to build, can be 1...n (unigram, bigram,)
    :param remove_stopwords: bool=Should stop words (english dictionary) be removed
    :param remove_numbers: bool=Should isolated numbers (12, 5.1) be removed
    :param remove_punctuation: bool=Should punctuation symbols be removed
    :return: a function accepting one parameter - str=document
    """
    def tokenize(document):
        """
        A custom tokenizer that is used by sklearn vectorizer functions. It is called on a complete corpus, which
        may depending on your case be the title of a document, a whole sentence or a complete document.
        :param document: str= the document to process
        :return: the collection of tokens as a vector or String
        """
        punctuation_to_strip = set(string.punctuation)
        words = word_tokenize(text=document, language='english')

        # substitute special words like ... analytic hierarchy method ...

        # remove punctuation
        if remove_punctuation and words.__len__() > 0:
            words = [w.strip() for w in words if w not in punctuation_to_strip]
        # remove numbers
        if remove_numbers and words.__len__() > 0:
            words = [w for w in words if re.match("^\D+$", w)]
        # remove stop words
        if remove_stopwords and words.__len__() > 0:
            words = [w for w in words if w not in stopwords.words('english')]

        if words.__len__() < 1:
            return []
        else:
            return ngrams(words, ngram)

    return tokenize


def fn_tokenize_unigram(remove_stopwords=True, remove_numbers=True, remove_punctuation=True):
    # convenience function
    return fn_tokenize_word(ngram=1, remove_stopwords=True, remove_numbers=True, remove_punctuation=True)


def fn_tokenize_bigram(remove_stopwords=True, remove_numbers=True, remove_punctuation=True):
    # convenience function
    return fn_tokenize_word(ngram=2, remove_stopwords=True, remove_numbers=True, remove_punctuation=True)


def word_vector_from_dictionary(dictionary, token, verbose=False):
    """
    Extract the word vector of the provided token from the dictionary, i.e. a one-hot encoding of the token
    in the space of vocabulary.
    :param dictionary:
    :param token:
    :param verbose:
    :return: the derived word vector, i.e. [1 0 0 0 0] for a vocabulary with 5 words, where the provided token
             corresponds to the first word in the dictionary
    """
    assert(isinstance(dictionary, dict))
    if "dictionary-length" not in dictionary:
        raise ValueError('word_vector_from_dictionary - dictionary does not have dictionary-length field')
    if "token" not in dictionary:
        raise ValueError('word_vector_from_dictionary - dictionary does not have token field')

    d = dictionary['dictionary-length']
    v = np.zeros(shape=(d, ), dtype=np.float32)
    idx = dictionary['token'].index(token)
    v[idx] = 1.0
    return v


def dictionary_to_word_vectors(dictionary, verbose = False):
    assert isinstance(dictionary, dict)

    word_vectors = dict()
    for a_token in dictionary['token']:
        if verbose:
            print("Processing token: %s" % a_token)

        if not word_vectors.has_key(a_token):
            v = word_vector_from_dictionary(dictionary=dictionary, token=a_token, verbose=verbose)
            word_vectors[a_token] = v
    return word_vectors


def dictionary_to_unigramtable(dictionary, unigram_table_size, verbose = False):
    assert(isinstance(dictionary, dict))
    assert(isinstance(unigram_table_size, int))

    if 'token' not in dictionary:
        raise ValueError('dictionary_to_unigramtable - token not present in dictionary')
    if 'token-counts' not in dictionary:
        raise ValueError('dictionary_to_unigramtable - token-counts not present dictionary')

    def _sample_word(voc_keys):
        l = voc_keys.__len__()
        u = random.randrange(start=0, stop=l, step=1)
        return voc_keys[u]

    toks = dictionary['token']
    toks_cnt = dictionary['token-counts']

    unigram_table = np.zeros(unigram_table_size)
    p_s = dict((k, round(toks_cnt[k] * unigram_table_size)) for k in toks)

    for i in range(unigram_table_size):
        w = _sample_word(toks)

        if p_s[str(w)] > 0:
            unigram_table[i] = w
            p_s[w] = p_s[w] - 1

    return unigram_table


def word_vector_corpus_to_matrix(dictionary, word_vectors, verbose):
    assert isinstance(dictionary, dict)
    assert isinstance(word_vectors, dict)

    l = []
    tokens = dictionary['token']
    corpus_len = dictionary['corpus-length']
    word_vector_dim = dictionary['dictionary-length']
    for idx_token in dictionary['indexed-corpus']:
        l.append(word_vectors[tokens[idx_token]])

    return np.array(l, dtype=np.float32).reshape(corpus_len, word_vector_dim)


def build_dictionary(corpus,
                     remove_stopwords=False, remove_numbers=False, remove_punctuation=False,
                     lower_case=True,
                     replace_dictionary=None,
                     verbose=True):
    """
    Build the dictionary structure hashtable from the provided corpus. While creating the structure a variety of
    transformations can be enacted, such as removing stop words are replacing parts of the corpus to something
    using a dictionary.
    :param corpus: (str)
    :param remove_stopwords: (bool)
    :param remove_numbers: (bool)
    :param remove_punctuation: (bool)
    :param lower_case: (bool)
    :param replace_dictionary: (dict)
    :param verbose: (bool)
    :return:
    """
    fn_corpus_preprocess = fn_preprocess_corpus(replace_dict=replace_dictionary,
                                                verbose=verbose,
                                                save_processed=None)

    fn_corpus_tokenize   = fn_tokenize_word(ngram=1,
                                            remove_stopwords=remove_stopwords,
                                            remove_numbers=remove_numbers,
                                            remove_punctuation=remove_punctuation)

    # this returns an ngram instance

    n_grams = _fn_build_dictionary(corpus = corpus,
                                   lower_case=lower_case,
                                   corpus_preprocessor=fn_corpus_preprocess,
                                   corpus_tokenizer=fn_corpus_tokenize,
                                   verbose=verbose)

    n_grams = list([t[0] for t in n_grams])
    unique_n_grams = list(set(n_grams))
    t_len = unique_n_grams.__len__()
    l_inv = 1. / t_len

    cnt_n_grams = Counter(n_grams)
    f_n_grams = [cnt_n_grams[t] * l_inv for t in unique_n_grams]
    indexed_corpus = [unique_n_grams.index(t) for t in n_grams]

    def _to_ngrams (str_text):
        assert(isinstance(str_text, str))

        n_c = _fn_build_dictionary(corpus=str_text,
                                   lower_case=lower_case,
                                   corpus_preprocessor=fn_corpus_preprocess,
                                   corpus_tokenizer=fn_corpus_tokenize,
                                   verbose=verbose)
        return list(a_tuple[0] for a_tuple in n_c)

    return {
        'token': unique_n_grams,
        'token-counts': cnt_n_grams,
        'token_frequencies' : f_n_grams,
        'dictionary-length': t_len,
        'corpus': corpus,
        'corpus-length': indexed_corpus.__len__(),
        'indexed-corpus': indexed_corpus,
        'to_ngrams': (lambda c: _to_ngrams(c)),
        'index_in_vocabulary': (lambda t: unique_n_grams.index(t)),
        'get_word-frequency' : (lambda t: f_n_grams[unique_n_grams.index(t)]),
        'get_word' : (lambda i: unique_n_grams[i])
    }


def cooccurrence_matrix_from_dictionary(dictionary, truncate_at = 0, verbose=False):
    """
    Build the symetrical token-co-occurrence matrix from the tokens provided in the dictionary. You can build the
    full count or truncate them at a positive integer value, if desired.
    :param dictionary: (dict) the dictionary structure providing the underlying corpus.
    :param truncate_at: (int) an integer you can use to truncate the co-occurrence counts (values > 0).
    :param verbose: (bool) should verbose logging be used.
    :return: the symetrical co-occurrence matrix. A numpy 2d matrix of type np.uint32.
    """
    assert(isinstance(dictionary, dict))

    if 'token' not in dictionary:
        raise ValueError('cooccurrence_matrix_from_dictionary - dictionary does not contain the token attribute.')
    if 'dictionary-length' not in dictionary:
        raise ValueError('cooccurrence_matrix_from_dictionary - dictionary does not contain the dictionary-length attribute.')
    if 'indexed-corpus' not in dictionary:
        raise ValueError('cooccurrence_matrix_from_dictionary - dictionary does not contain the indexed-corpus attribute.')

    no_toks = dictionary['dictionary-length']
    idx_corpus = dictionary['indexed-corpus']
    l          = idx_corpus.__len__()
    l          = int(math.ceil(l / 2))
    cooaccurrence_matrix = np.zeros(shape=(no_toks, no_toks), dtype=np.uint32)

    truncate_fn = lambda x: x

    if truncate_at > 0:
        truncate_fn = lambda x: truncate_at if x > truncate_at else x

    # fill the symetric matrix
    for i in range(1, l):
        j = i + 1
        t_i = idx_corpus[i]
        t_j = idx_corpus[j]
        cooaccurrence_matrix[t_i][t_j] = truncate_fn(cooaccurrence_matrix[t_i][t_j] + 1)
        cooaccurrence_matrix[t_j][t_i] = cooaccurrence_matrix[t_i][t_j]

    return cooaccurrence_matrix


def cooccur_wordij(dictionary, cooccur_matrix, wordi, wordj):
    assert(isinstance(dictionary, dict))
    assert(isinstance(wordi, str))
    assert(isinstance(wordj, str))

    if 'index_in_vocabulary' not in dictionary:
        raise ValueError('cooccur_wordij - dictionary does not define index_in_vocabulary.')

    t_i = dictionary['index_in_vocabulary'](wordi)
    t_j = dictionary['index_in_vocabulary'](wordj)

    return cooccur_matrix[t_i][t_j]


def __negative_sample(dictionary, word, s_token_frequencies):
    #    P(w) = f(w)^.75 / sum(j) (f(w_j))^.75
    assert(isinstance(dictionary, dict))
    assert(isinstance(word, str))

    if 'get_word-frequency' not in dictionary:
        raise ValueError('negative_sample - get_word-frequency not in dictionary')
    # get the frequency of the word
    gwf = dictionary['get_word-frequency']
    f_w = gwf(word)

    s_f_tokens = f_w**.75 / s_token_frequencies
    return s_f_tokens


def negative_sample(dictionary, word):
    assert(isinstance(dictionary, dict))
    assert(isinstance(word, str))
    if 'token_frequencies' not in dictionary:
        raise ValueError('negative_sample - token_frequencies not in dictionary')

    tok_frequencies = [t_f**.75 for t_f in dictionary['token_frequencies']]
    s_tok_frequencies = np.sum(tok_frequencies)

    return __negative_sample(dictionary=dictionary, word=word, s_token_frequencies=s_tok_frequencies)


def _fn_build_dictionary(corpus,
                         lower_case=True,
                         corpus_preprocessor = None,
                         corpus_tokenizer    = None,
                         verbose = True):

    if verbose:
        print("fn_build_dictionary:")
        print("lower case: %s" % lower_case)
        print("pre-process: %s" % (corpus_preprocessor is not None))
        print("tokenize:%s" % (corpus_tokenizer is not None))

    pre_processed_corpus = corpus
    if lower_case:
        pre_processed_corpus = corpus.lower()

    if corpus_preprocessor:
        assert isinstance(pre_processed_corpus, str)
        pre_processed_corpus = corpus_preprocessor(document=pre_processed_corpus)

    if corpus_tokenizer:
        pre_processed_corpus = corpus_tokenizer(document=pre_processed_corpus)

    # this returns an ngram instance
    return pre_processed_corpus


def build_previous_wordcorpus_from_dictionary(dictionary, n_previous_words = 2):
    assert(isinstance(dictionary, dict))

    if 'indexed-corpus' not in dictionary:
        raise ValueError('build_previous_wordcorpus_from_dictionary - indexed-tokens not defined in dictionary')

    indexed_tokens = dictionary['indexed-corpus']
    l = indexed_tokens.__len__()
    v_data = []
    for i_start in range(n_previous_words, l):
        v_previous = [indexed_tokens[idx_j] for idx_j in range(i_start - n_previous_words, i_start)]
        v = indexed_tokens[i_start]
        v_data.append((v, v_previous))

    return v_data


def pcond_ij(cooccurrence_matrix, token_idx_i, token_idx_j):
    """
    P_ij = P(i|j) = p_i_and_j / p_j
    :param cooccurrence_matrix:
    :param token_idx_i:
    :param token_idx_j:
    :return:
    """
    c_j = np.sum(cooccurrence_matrix[token_idx_j, :])
    c_i_and_j = cooccurrence_matrix[token_idx_j, token_idx_i]
    return c_i_and_j / c_j


def replace_using_dictionary(text, replace_dict):
    assert (isinstance(replace_dict, dict))

    if not text or not text.strip:
        raise ValueError("replace_using_dictionary - text missing")

    out_text = text.strip()
    if len(replace_dict) > 0:
        for key in replace_dict.keys():
            find_pos = out_text.find(key)
            if find_pos >= 0:
                val = replace_dict[key]
                out_text = out_text.replace(key, val)

    return out_text
