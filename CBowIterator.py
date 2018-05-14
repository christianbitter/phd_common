import mxnet as mx
from mxnet import nd
import math
import numpy as np

# in the cbow task, our goal is to predict a context center word, provided we have access to the
# context ... so our label shapes are actually (n, 1)
#             so our data shapes are actually (n, 2 * context size)

class CBowIterator(mx.io.DataIter):
    def __init__(self, dictionary_corpus, no_context_words, batch_size=5, verbose=False):
        super(CBowIterator, self).__init__(batch_size=batch_size)

        self._verbose = verbose
        self._provide_label = list(zip(['context_words'], [(batch_size, 2 * no_context_words)]))
        self._provide_data = list(zip(['center_word'], [(batch_size, )]))
        self.batch_size = batch_size
        # from the corpus generate the respective data and labels
        self.cbow_corpus = CBowIterator.__build_cbow_data(document=dictionary_corpus, context_size=no_context_words)
        self.label_gen = [mx.nd.array(context_word_vec).astype(dtype=np.int32) for context_word_vec, _ in self.cbow_corpus]
        self.data_gen  = [center_word for _, center_word in self.cbow_corpus]
        self.cur_batch = -1
        # the number of batches is the number of individuals divided by batch size + 1
        no_individuals   = self.cbow_corpus.__len__()
        self.num_batches = math.ceil(no_individuals / batch_size)

    @staticmethod
    def __build_cbow_data(document, context_size=2):
        if document is None:
            ValueError('__build_cbow_data - document is None')

        data, individual = [], []
        l_i = (2 * context_size + 1)
        for i in range(document.__len__() - l_i + 1):
            individual = []
            for j in range(l_i):
                i_j = i + j
                if j != context_size:
                    individual.append(document[i_j])
            data.append((individual, document[i + context_size]))

        return data

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = -1

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def iter_next(self):
        self.cur_batch += 1

        if self.cur_batch < self.num_batches:
            return True
        else:
            return False

    def getdata(self):
        start_index = self.cur_batch * self.batch_size
        end_index   = min(start_index + self.batch_size, self.data_gen.__len__)

        return self.data_gen[start_index:end_index]

    def getlabel(self):
        start_index = self.cur_batch * self.batch_size
        end_index   = min(start_index + self.batch_size, self.data_gen.__len__)

        return self.label_gen[start_index:end_index]

    def getindex(self):
        return self.cur_batch

    @property
    def number_of_batches(self):
        return self.num_batches
