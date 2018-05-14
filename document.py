import os
import csv
import content_block
from phd_common import Corpus


class Document(object):
    def __init__(self):
        super(Document, self).__init__()


class IMDocument(Document):
    def __init__(self, content_blocks):
        super(IMDocument, self).__init__()
        assert isinstance(content_blocks, list)

        self.content_blocks = content_blocks
        self._title = None
        self._abstract = None
        self._keywords = None
        self._document = {}

        self.__process_content_blocks()

    @staticmethod
    def __depth_chapter(chapter, d):
        assert (isinstance(chapter, str))
        assert (isinstance(d, int))

        if chapter == "chapter":
            return d + 1
        else:
            if chapter.startswith("sub"):
                return IMDocument.__depth_chapter(chapter[3:], d + 1)
            else:
                raise ValueError("__depth_chapter - unknown %s" % chapter)

    def __process_content_blocks(self):

        content_types = {'title': 'title',
                         'abstract': 'abstract',
                         'keyword': 'keyword',
                         'document': 'document'}
        active_chapter = None
        order = 0

        for cb in self.content_blocks:
            cb_type = cb.keys()[0]
            cb_xtype = content_types.get(cb_type, 'document')
            cb_content = cb[cb_type]

            if cb_xtype == 'title':
                self._title = cb_content
            elif cb_xtype == 'abstract':
                self._abstract = cb_content
            elif cb_xtype == 'keyword':
                self._keywords = cb_content
            else:
                # content will be associated with the chapter it came from
                if cb_type == "content":
                    if active_chapter is None:
                        raise ValueError('Cannot associate content with active_chapter - None')
                    else:
                        self._document[active_chapter]['content'] = self._document[active_chapter][
                                                                        'content'] + cb_content
                else:
                    active_chapter = cb_content
                    self._document[active_chapter] = {
                        'content': "",
                        'order': order,
                        'depth': IMDocument.__depth_chapter(cb_type, 0)
                    }
                    order = order + 1

        return None

    @property
    def title(self):
        return self._title

    @property
    def abstract(self):
        return self._abstract

    @property
    def keywords(self):
        return self._keywords

    @property
    def document(self):
        return self._document


def intermediate_csv_to_imdocument(csv_fp, encoding='utf-8', to_ascii=True, verbose=False):
    assert isinstance(csv_fp, str)
    if not csv_fp:
        raise ValueError("intermediate_csv_to_imdocument - csv_fp cannot be None or empty")

    if not os.path.isfile(csv_fp):
        raise ValueError("intermediate_csv_to_imdocument - csv_fp does not point to valid intermediate csv")

    if verbose:
        print("intermediate_csv_to_imdocument - reading %s" % csv_fp)

    blocks = content_block.intermediate_csv_to_content_block(csv_fp=csv_fp, encoding=encoding, to_ascii=to_ascii, verbose=verbose)

    if verbose:
        print("intermediate_csv_to_imdocument - read %s no. content blocks" % len(blocks))

    return IMDocument(content_blocks=blocks)


def linearize_imdocument_to_tokens(im_document, options, verbose=False):
    assert (isinstance(im_document, IMDocument))
    assert (isinstance(options, dict))

    # we get all the content in order and create a corpus
    # from that we get the individual tokens
    c = ""

    if im_document.title:
        c = c + " " + im_document.title

    if im_document.abstract:
        c = c + " " + im_document.abstract

    if im_document.keywords:
        c = c + " " + im_document.keywords

    sorted_content_blocks = im_document.document
    # this returns a list of the sorted keys ... so we can index in order the content blocks in the
    # document.
    sorted_content_blocks = sorted(sorted_content_blocks,
                                   key=lambda content_block: sorted_content_blocks[content_block]['order'])

    for a_content_block in sorted_content_blocks:
        if verbose:
            print("Content-Block: %s" % a_content_block)

        b = im_document.document[a_content_block]
        c = c + " " + a_content_block
        c = c + " " + b['content']

    remove_stopwords = options.get('remove_stopwords', False)
    remove_numbers = options.get('remove_numbers', False)
    remove_punctuation = options.get('remove_punctuation', False)
    lower_case = options.get('lower_case', True)
    replace_dictionary = options.get('replace_dictionary', None)

    corpus = Corpus.Corpus.build_corpus(corpus=c,
                                        lower_case=lower_case,
                                        remove_numbers=remove_numbers,
                                        remove_stopwords=remove_stopwords,
                                        remove_punctuation=remove_punctuation,
                                        replace_dictionary=replace_dictionary)

    return corpus.corpus_tokenized()


def imtokens_to_csv(tokens, csv_fp, options, verbose=False):
    assert (isinstance(tokens, list))
    assert (isinstance(csv_fp, str))
    assert (isinstance(options, dict))

    dir_fp = os.path.dirname(csv_fp)

    if not os.path.isdir(dir_fp):
        raise ValueError("imtokens_to_csv - '%s' does not exist" % dir_fp)

    token_dict = [{'token': t} for t in tokens if t.strip() != ""]

    delimiter = options.get('delimiter', ';')
    quotechar = options.get('quotechar', '"')
    write_header = options.get('write_header', True)

    with open(csv_fp, 'wb') as csvfile:
        token_writer = csv.DictWriter(csvfile,
                                      delimiter=delimiter,
                                      quotechar=quotechar, quoting=csv.QUOTE_MINIMAL, fieldnames=['token'])
        if write_header:
            token_writer.writeheader()

        token_writer.writerows(token_dict)

    return None


def read_csv_tokens(csv_fp, options, verbose=False):
    assert (isinstance(csv_fp, str))
    assert (isinstance(options, dict))

    if not os.path.isfile(csv_fp):
        raise ValueError("read_csv_tokens - '%s' does not exist" % csv_fp)

    corpus_tokens = []

    with open(csv_fp, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['Token'])
        file_tokens = [t['Token'] for t in reader]
        corpus_tokens.extend(file_tokens)

    if verbose:
        print("read_csv_tokens: %s has %d tokens" % (csv_fp, len(corpus_tokens)))

    return corpus_tokens
