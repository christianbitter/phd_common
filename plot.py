import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from numpy.linalg import svd
from scipy.cluster.hierarchy import dendrogram, linkage

def plot_wordcloud_from_tokencount_pairs(tokencounts, do_plot=True):
    dict_tokencount = dict(tokencounts)
    wordcloud = WordCloud().generate_from_frequencies(dict_tokencount)
    if do_plot:
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    return wordcloud


def plot_word_histogram(vectorizer, bag_of_words, show_plot = True, topN=20, bar_width = 0.35):
    """
    Create and show bar plot of occurrences of words from the bag of words. Note: the topN option does not apply
    to the returned label-count vector. This returns the full data always.
    :param vectorizer: CountVectorizer=sklearn count vectorizer to be used to determine the feature names
    :param bag_of_words: csr_matrix=bag of words derived from CountVectorizer fit call
    :param topN: int=Threshold of words to plot
    :param bar_width: float=Width of bar in the plot. Controls the placement of the word label under it's occurence bar.
    :return: the word-frequency pairs
    """
    word_list = vectorizer.get_feature_names()

    # the index of words into the bow representing is vectorizer.vocabulary_.get(<term>)
    counts = bag_of_words.toarray()
    word_counts = np.sum(a=counts, axis=0)

    label_values = [[word, word_counts[word_list.index(word)]] for word in word_list]
    labels, values = zip(*label_values)

    # sort your values in descending order
    indSort = np.argsort(values)[::-1]

    # rearrange your data
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]
    indexes = np.arange(len(labels))

    if show_plot:
        t_idx = indexes
        t_val = values
        t_lab = labels

        if topN > 0:
            t_idx = indexes[:topN]
            t_val = values[:topN]
            t_lab = labels[:topN]

        plt.bar(t_idx, t_val)
        # add labels
        plt.xticks(t_idx + bar_width, t_lab)
        if topN > 0:
            plt.title("Occurrence of top-%s words" % topN)
        else:
            plt.title("Occurrence of words")
        plt.show()

    return label_values


def text_point_plot(vocab, coordinate_matrix, title, b=.2, verbose=False):
    assert(isinstance(vocab, dict))

    if 'dictionary-length' not in vocab:
        raise ValueError('text_point_plot - vocab does not define dictionary-length')

    if 'get_word' not in vocab:
        raise ValueError('text_point_plot - vocab does not define get_word')#

    x = coordinate_matrix[:, 0]
    y = coordinate_matrix[:, 1]
    xr = [np.min(x), np.max(x)]
    yr = [np.min(y), np.max(y)]

    plt.title(title)
    plt.xlim([xr[0] - b * (xr[1] - xr[0]), xr[1] + b * (xr[1] - xr[0])])
    plt.ylim([yr[0] - b * (yr[1] - yr[0]), yr[1] + b * (yr[1] - yr[0])])

    for j in range(vocab['dictionary-length']):
        w = vocab['get_word'](j)
        plt.text(x[j], y[j], w, withdash=True)

    plt.show()

def plot_svd_of_cooccurrence_matrix(vocab, cooccurrence_matrix, b=.2, verbose=False):
    assert(isinstance(vocab, dict))

    if 'dictionary-length' not in vocab:
        raise ValueError('plot_svd_of_cooccurrence_matrix - vocab does not define dictionary-length')

    if 'get_word' not in vocab:
        raise ValueError('plot_svd_of_cooccurrence_matrix - vocab does not define get_word')

    U, s, vh = svd(cooccurrence_matrix, full_matrices=True, compute_uv=True)

    text_point_plot(vocab=vocab, coordinate_matrix=U,
                    title = 'Plot of first two SVD coordinates of corpus co-occurrence matrix',
                    b=b,
                    verbose=verbose)

    return None


def plot_dendogram(cooccurence_matrix, tokens, orientation='left', verbose=False):
    Z = linkage(cooccurence_matrix)
    dendrogram(Z, above_threshold_color='y', orientation=orientation, labels=tokens)
    plt.title("Hierarchical Clustering/ Dendogram Plot of word co-occurrence")
    plt.show()
