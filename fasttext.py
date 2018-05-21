import io
import os
from subprocess import call

import misc


class FastText(object):
    """
    A simple wrapper around the pre-compiled fasttext utility.

    The commands supported by fasttext are:

    supervised              train a supervised classifier
    quantize                quantize a model to reduce the memory usage
    test                    evaluate a supervised classifier
    predict                 predict most likely labels
    predict-prob            predict most likely labels with probabilities
    skipgram                train a skipgram model
    cbow                    train a cbow model
    print-word-vectors      print word vectors given a trained model
    print-sentence-vectors  print sentence vectors given a trained model
    print-ngrams            print ngrams given a trained model and word
    nn                      query for nearest neighbors
    analogies               query for analogies
    dump                    dump arguments, dictionary, input / output vectors
    """

    FT_SUCCESS = 0

    CMD_PRINT_SENTENCE_VECTORS = "print-sentence-vectors"
    CMD_PRINT_WORD_VECTORS = "print-word-vectors"

    def __init__(self, ft_path_fp, remove_intermediate_files=False, **kwargs):
        super(FastText, self).__init__(**kwargs)
        self.cmd = ft_path_fp
        self.ft_dir_fp = os.path.dirname(self.cmd)
        self.remove_intermediate_files = remove_intermediate_files

    def supervised(self):
        pass

    def quantize(self):
        pass

    def test(self):
        pass

    def predict(self, model_fp, test_data_fp, k = None, threshold=None):
        """
        fasttext predict[-prob] <model> <test-data> [<k>] [<th>]
        :param model_fp: (str) path to the model's bin file
        :param test_data_fp: (str) either path to text file containing testing/ prediction data or a string to predict.
        :param k: (int) number of labels to return
        :param threshold: (threshold) threshold of the prediction
        :return: (list) returns a list (possibly empty) of predictions for the input. The __label__ identifier is
        removed for improved readability.
        """
        assert (isinstance(model_fp, str))
        assert (isinstance(test_data_fp, str))
        assert (isinstance(k, int))

        if not model_fp:
            raise ValueError("predict - model_fp not provided")

        if not test_data_fp:
            raise ValueError("predict - test_data_fp not provided (can be filepath or string)")

        if not os.path.isfile(model_fp):
            raise ValueError("predict - model_fp not provided")

        args = [self.cmd,
                "predict",
                model_fp]

        if test_data_fp is not None:
            if os.path.isfile(test_data_fp):
                args.extend([test_data_fp])
            else:
                # for now simulate the stdin with a file
                predict_in_fp = misc.get_temp_file(".txt")
                with io.open(predict_in_fp, 'wb+') as in_file:
                    in_file.write(test_data_fp)
                args.extend([predict_in_fp])

        if k is not None:
            args.extend([str(k)])

        if threshold is not None:
            args.extend([threshold])

        predict_out_fp = misc.get_temp_file()
        return_code = call(args=args, shell=False, cwd=self.ft_dir_fp, stdout=io.open(predict_out_fp, 'w+'))

        prediction = []

        if return_code == FastText.FT_SUCCESS:
            with io.open(file=predict_out_fp, mode='r+') as info_file:
                s_prediction = info_file.readlines()
                prediction = []
                for l in s_prediction:
                    v_prediction = [p.replace("__label__", "") for p in l.split(" ")]

                    prediction.append(v_prediction)

        return prediction

    def predict_prob(self):
        pass

    def skipgram(self, input_fp, output_fp):
        assert (isinstance(input_fp, str))
        assert (isinstance(output_fp, str))

        if not input_fp:
            raise ValueError("skipgram - input_fp not provided")
        if not input_fp:
            raise ValueError("skipgram - output_fp not provided")
        if not os.path.isfile(input_fp):
            raise ValueError("skipgram - input_fp does not exist")
        out_dir_fp = os.path.dirname(output_fp)
        if not os.path.isdir(out_dir_fp):
            raise ValueError("skipgram - output directory does not exist")

        args = [self.cmd,
                "skipgram",
                "-input", input_fp,
                "-output", output_fp]

        return_code = call(args=args, shell=False, cwd=self.ft_dir_fp)
        return return_code == 0

    def cbow(self):
        pass

    def print_word_vectors(self, model_fp, words):
        """
        Prints or returns the word vectors for the provided choice of words
        :param model_fp: (str) file path to fasttext trained model bin
        :param words: (str) string of desired tokens
        :return: (dict) a dictionary with keys being tokens and values the individual word vectors
        """
        assert (isinstance(model_fp, str))
        assert (isinstance(words, str))
        if not model_fp:
            raise ValueError("print_word_vectors - path to model vec file not defined.")

        if not os.path.isfile(model_fp):
            raise ValueError("print_word_vectors - not a vec file.")

        word_vectors = {}
        if words:
            word_fp = misc.get_temp_file(".txt")
            with io.open(word_fp, "w+", encoding="ascii") as f_words:
                f_words.write(words.decode("utf-8"))

            args = [self.cmd,
                    FastText.CMD_PRINT_WORD_VECTORS,
                    model_fp]

            word_vector_fp = misc.get_temp_file()

            return_code = call(args=args, shell=False, cwd=self.ft_dir_fp,
                               stdin=io.open(word_fp, 'r+'),
                               stdout=io.open(word_vector_fp, 'w+'))

            if return_code == FastText.FT_SUCCESS:
                print(word_vector_fp)
                with io.open(word_vector_fp, "r+") as f_wordvectors:
                    for a_vector in f_wordvectors.readlines():
                        a_line   = a_vector.split(" ")
                        token    = a_line[0]
                        a_vector = [float(s) for s in a_line[1:] if not s.isspace()]
                        word_vectors[token] = a_vector

            # cleanup
            if self.remove_intermediate_files:
                os.remove(word_fp)
                os.remove(word_vector_fp)

        return word_vectors

    def print_sentence_vectors(self, model_fp, sentence):
        assert (isinstance(model_fp, str))
        assert (isinstance(sentence, str))
        if not model_fp:
            raise ValueError("print_sentence_vectors - path to model vec file not defined.")

        if not os.path.isfile(model_fp):
            raise ValueError("print_sentence_vectors - not a vec file.")

        sentence_vector = {}
        if sentence:
            sentence_fp = misc.get_temp_file(".txt")
            with io.open(sentence_fp, "w+", encoding="ascii") as f_sentence:
                f_sentence.write(sentence.decode("utf-8"))

            args = [self.cmd,
                    FastText.CMD_PRINT_SENTENCE_VECTORS,
                    model_fp]

            sentence_vector_fp = misc.get_temp_file()

            return_code = call(args=args, shell=False, cwd=self.ft_dir_fp,
                               stdin=io.open(sentence_fp, 'r+'),
                               stdout=io.open(sentence_vector_fp, 'w+'))

            if return_code == FastText.FT_SUCCESS:
                with io.open(sentence_vector_fp, "r+") as f_svectors:
                    for a_vector in f_svectors.readlines():
                        a_line = a_vector.split(" ")
                        token = a_line[0]
                        a_vector = [float(s) for s in a_line[1:] if not s.isspace()]
                        sentence_vector[token] = a_vector

            # cleanup
            if self.remove_intermediate_files:
                os.remove(sentence_fp)
                os.remove(sentence_vector_fp)

        return sentence_vector

    def print_ngrams(self):
        pass

    def nn(self):
        pass

    def analogies(self):
        pass

    def dump(self):
        pass