import io
import os
import tempfile
import uuid
from subprocess import call


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
    def __init__(self, ft_path_fp, **kwargs):
        super(FastText, self).__init__(**kwargs)
        self.cmd = ft_path_fp
        self.ft_dir_fp = os.path.dirname(self.cmd)

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
                predict_in_fp = os.path.join(tempfile.gettempdir(), "%s" % uuid.uuid4())
                with io.open(predict_in_fp, 'wb+') as in_file:
                    in_file.write(test_data_fp)
                args.extend([predict_in_fp])

        if k is not None:
            args.extend([str(k)])

        if threshold is not None:
            args.extend([threshold])

        predict_out_fp = os.path.join(tempfile.gettempdir(), "%s" % uuid.uuid4())
        return_code = call(args=args, shell=False, cwd=self.ft_dir_fp, stdout=io.open(predict_out_fp, 'w+'))

        prediction = []

        if return_code == 0:
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

    def print_word_vectors(self):
        pass

    def print_sentence_vectors(self):
        pass

    def print_ngrams(self):
        pass

    def nn(self):
        pass

    def analogies(self):
        pass

    def dump(self):
        pass