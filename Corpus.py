from phd_common.nlp import cooccurrence_matrix_from_dictionary, build_dictionary

class Corpus(object):
    """
    The corpus type provides a more accessible method to interact with the functionality already provided in the nlp package and dictionary abstraction.
    """

    def __init__(self, dictionary):
        super(Corpus, self).__init__()
        self.dictionary = dictionary

    @classmethod
    def build_corpus(cls, corpus,
                     remove_stopwords=False, remove_numbers=False, remove_punctuation=False,
                     lower_case=True,
                     replace_dictionary=None,
                     verbose=True):
        assert(isinstance(corpus, str))
        d = build_dictionary(corpus=corpus,
                             remove_stopwords=remove_stopwords,
                             remove_numbers=remove_numbers,
                             remove_punctuation=remove_punctuation,
                             lower_case=lower_case,
                             replace_dictionary=replace_dictionary,
                             verbose=verbose)
        return Corpus(d)

    @property
    def as_dictionary(self):
        """
        return the dictionary structured from which we built up this Corpus object
        :return: (dict) dictionary representing the corpus
        """
        return self.dictionary

    @property
    def corpus_length(self):
        return self.dictionary['corpus-length']

    @property
    def dictionary_length(self):
        return self.dictionary['dictionary-length']

    @property
    def token(self):
        return self.dictionary['token']

    @property
    def token_counts(self):
        return self.dictionary['token-counts']

    @property
    def token_frequencies(self):
        return self.dictionary['token_frequencies']

    def word_index_in_vocabulary(self, word):
        return self.dictionary['index_in_vocabulary'](word)

    def get_word(self, token_index):
        if token_index < 0 or token_index > self.dictionary_length:
            raise ValueError("get_word - token index cannot be outside [0, no-tokens]")

        return self.dictionary['get_word'](token_index)

    def get_word_frequency(self, token):
        return self.dictionary['get_word-frequency'](token)

    @property
    def corpus(self):
        return self.dictionary['corpus']

    def build_cooccurrence_matrix(self, truncate_at = 0):
        if truncate_at < 0:
            raise ValueError('build_cooccurrence_matrix - truncate_at cannot be smaller 0')
        return cooccurrence_matrix_from_dictionary(self.dictionary, truncate_at = truncate_at, verbose=False)

    @property
    def indexed_corpus(self):
        return self.dictionary['indexed-corpus']

    def text_to_ngrams(self, text):
        if text is None:
            raise ValueError('text_to_ngrams - text cannot be none')

        return self.dictionary['to_ngrams'](text)

    def corpus_tokenized(self):
        indexed_corpus = self.indexed_corpus
        return [self.get_word(idx_token) for idx_token in indexed_corpus]
