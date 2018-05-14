from subprocess import call
import os
from os import path
import tempfile
import shutil
import uuid
import types
import glob
import pdf

from bs4 import BeautifulSoup


def outline_to_im(pdf_fp,
                  fp_dir_pdf2htmlEx="C:\\pdf2htmlEX",
                  verbose=False):
    """
    Provided the path to a pdf file, we use pdf2htmlEx to generate an outline file. This outline file does only
    contain the structure of the document. This structure is subsequently used to generate an intermediate form.
    :param pdf_fp: (str) path to pdf document
    :param fp_dir_pdf2htmlEx: (str) path to pdf2htmlex programm
    :param verbose: (bool) should verbose printing be used
    :return:
    """
    assert (isinstance(pdf_fp, str))
    assert (isinstance(fp_dir_pdf2htmlEx, str))

    if not os.path.isdir(fp_dir_pdf2htmlEx):
        raise ValueError("outline_pdf2html - pdf2htmlex does not exist?")

    if not pdf_fp:
        raise ValueError("outline_pdf2html - pdf file does not exist?")

    if not os.path.isfile(pdf_fp):
        raise ValueError("outline_pdf2html - pdf file does not exist?")

    # create the outline file into a tempfile, read it back and create the im file
    outline_fp = outline_pdf2html(pdf_fp=pdf_fp,
                                  fp_dir_pdf2htmlEx=fp_dir_pdf2htmlEx,
                                  copy_to_outline_file=False,
                                  verbose=verbose)

    if outline_fp:
        if not os.path.isfile(outline_fp):
            raise ValueError('outline_to_im - outline2html generated outline file but this cannot be found')

        parsed_html = None
        with open(outline_fp, 'r+') as f_outline:
            parsed_html = BeautifulSoup(f_outline, 'html.parser')
        return parsed_html
    else:
        return None


def outline_pdf2html(pdf_fp,
                     fp_dir_pdf2htmlEx="C:\\pdf2htmlEX",
                     fp_outdir="C:\\CloudStation\\CBitter_-_PHD\\CloudStation\\PHDLiterature\\AHP_scopus\\html",
                     outline_file_name="outline.html",
                     copy_to_outline_file=True,
                     verbose=False):
    """
    Using pdf2htmlex, we create the outline file. That is, a file providing us with chapter structure
    of the pdf document. This will give (if the underlying pdf is reasonably formated) title, chapter, subchapter, ...
    Respectively, we can use this information to place content in the intermediate format and process it.
    :param pdf_fp: (str) file path to pdf file.
    :param fp_dir_pdf2htmlEx: (str) path to the pdf2htmlEx executable.
    :param fp_outdir: (str) path to directory where outline file is to be generated.
    :param outline_file_name: (str) name of outline file
    :param copy_to_outline_file:
    :param verbose: (boolean) should verbose logging be used.
    :return:
    """
    assert (isinstance(pdf_fp, str))
    assert (isinstance(fp_outdir, str))
    assert (isinstance(fp_dir_pdf2htmlEx, str))
    assert (isinstance(outline_file_name, str))

    if not os.path.isdir(fp_outdir):
        raise ValueError("outline_pdf2html - fp_outdir does not exist?")

    if not os.path.isdir(fp_dir_pdf2htmlEx):
        raise ValueError("outline_pdf2html - pdf2htmlex does not exist?")

    if not os.path.isfile(pdf_fp):
        raise ValueError("outline_pdf2html - pdf file does not exist?")

    if not outline_file_name:
        raise ValueError("outline_pdf2html - outline file name not provided")

    outline_fp = os.path.join(fp_outdir, outline_file_name)

    # we generate the content into a temp dir
    # then take the outline file and stuff its contents into the outline_file
    pdf_fname, _ = path.splitext(path.basename(pdf_fp))
    outline_fname= "%s.outline" % pdf_fname

    ret_code = pdf2html(fp_pdf=pdf_fp,
                        generated_html_filename="generated.html",
                        fp_outdir=fp_outdir,
                        splitPages=False,
                        optimizeText=False,
                        processNonText=True,
                        preprocessToPS=False,
                        options={
                            "process-outline": "1"
                        },
                        debug=verbose)

    tmp_outline_fp = os.path.join(fp_outdir, outline_fname)

    if copy_to_outline_file:
        shutil.copy(tmp_outline_fp, outline_fp)
    else:
        outline_fp = tmp_outline_fp

    if verbose:
        print("outline html file: %s (%s)" % (outline_fp, ret_code))

    if ret_code:
        return outline_fp
    else:
        return None


# As noted here, https://github.com/coolwanglu/pdf2htmlEX/issues/759
# it might improve stability to convert the text to ps and then to pdf before working the content
# pdf to ps and back via pdftops and pstopdf
def pdf2html(fp_pdf,
             fp_dir_pdf2htmlEx="C:\\pdf2htmlEX",
             fp_dir_pdf2utils ="C:\\Development\\xpdf-tools-win-4.00\\bin64",
             fp_dir_ps2pdf    ="C:/Program Files/gs/gs9.21/lib/",
             fp_outdir="C:\\CloudStation\\CBitter_-_PHD\\CloudStation\\PHDLiterature\\AHP_scopus\\html",
             generated_html_filename="result.html",
             generated_pagehtml_filename = "page_%d.html",
             splitPages=True,
             processNonText=False,
             optimizeText=True,
             preprocessToPS=True,
             options={},
             debug=False):

    assert (isinstance(fp_pdf, str))
    assert (isinstance(fp_outdir, str))
    assert (isinstance(fp_dir_pdf2htmlEx, str))
    assert (isinstance(generated_html_filename, str))
    assert (isinstance(generated_pagehtml_filename, str))
    assert (isinstance(options, dict))

    if not path.exists(fp_pdf):
        raise ValueError("pdf2html - file path ('%s') does not exist." % fp_pdf)

    if not path.isfile(fp_pdf):
        raise ValueError("pdf2html - file path to pdf document is not a file.")

    if not path.exists(fp_outdir):
        raise ValueError("pdf2html - file path to target directory does not exist.")

    if not path.exists(fp_dir_pdf2htmlEx):
        raise ValueError("pdf2html - file path to pdf2htmlEx does not exist.")

    pdf2html = os.path.join(fp_dir_pdf2htmlEx, "pdf2htmlEX.exe")
    pdf2ps   = os.path.join(fp_dir_pdf2utils, "pdftops.exe")
    ps2pdf   = os.path.join(fp_dir_ps2pdf, "ps2pdf.bat")

    if debug:
        print("pdf2html: %s" % pdf2html)
        print("pdf2ps  : %s" % pdf2ps)
        print("ps2pdf  : %s" % ps2pdf)

    file_name, _ = path.splitext(path.basename(fp_pdf))

    # As noted here, https://github.com/coolwanglu/pdf2htmlEX/issues/759
    # it might improve stability to convert the text to ps and then to pdf before working the content
    # ...  convert the pdf to ps and back via pdftops and pstopdf using the xpdf package.
    # get temp file name/ path
    if preprocessToPS:
        temp_dir_fp = tempfile.gettempdir()
        temp_ps_fp = path.join(temp_dir_fp, ("%s.ps" % tempfile.gettempprefix()))
        temp_pdf_fp= path.join(temp_dir_fp, ("%s.pdf" % tempfile.gettempprefix()))

        if debug:
            print("ps tempfile: %s" % temp_ps_fp)
            print("pdf tempfile: %s" % temp_pdf_fp)

        # convert pdf to ps
        r_pdf2ps = call(args=[pdf2ps, fp_pdf, temp_ps_fp], shell=True, cwd=fp_dir_pdf2utils)
        r_ps2pdf = call(args=[ps2pdf, temp_ps_fp, temp_pdf_fp], shell=True, cwd=fp_dir_ps2pdf)

        # convert ps to pdf and use input
        fp_pdf  = temp_pdf_fp

    # the generated content will be placed into the directory specified by fp_outdir
    # the user is responsible for creating the directory and cleaning up any mess
    html_fp = os.path.join(fp_outdir)

    if debug:
        print("html file: %s" % html_fp)

    pdf2html_args = [
        pdf2html, fp_pdf
    ]

    if not splitPages:
        pdf2html_args.extend([generated_html_filename])

    pdf2html_args.extend(["--dest-dir", html_fp])

    if debug:
        pdf2html_args.extend(["--debug", "1"])

    if splitPages:
        pdf2html_args.extend(["--split-pages", "1",
                              "--page-filename", generated_pagehtml_filename])

    if processNonText:
        pdf2html_args.extend(["--process-nontext", "1"])

    if optimizeText:
        pdf2html_args.extend(["--optimize-text", "1"])

    if debug:
        print pdf2html_args

    process_outline = options.get("process-outline", "0")
    support_printing= options.get("printing", "0")

    pdf2html_args.extend(
        [
            "--embed-font", "0",
            "--embed-javascript", "0",
            "--embed-image", "0",
            "--embed-outline", "0",
            # "--embed-css", "0",
            "--process-outline", process_outline,
            "--bg-format", "jpg",
            "--printing", support_printing
        ]
    )

    retcode = call(args=pdf2html_args,
                   shell=False,
                   cwd=fp_dir_pdf2htmlEx)

    return retcode == 0


def structure_from_bs4outline(pdf_fp,
                              verbose=False):
    """
    Given the path to a pdf file, we read the pdf using pdf2html and process it's outline file into a dictionary
    that provides us with an ordered (key) view on the individual chapters and their relationships to the document.
    :param pdf_fp: (str) path to pdf document.
    :param verbose: (bool) should intermediate information be printed.
    :return: (dict) the document in a content-block structured format.
    """
    def __destructure(bs4_html, level, seq_id, struct):
        # if we encounter a ul block we increment the level
        # if we encounter an end block we decrement
        # for each li we create a block
        t = bs4_html
        if t.name == 'ul':
            level = level + 1
            seq_id= 0
            for n in t.children:
                seq_id = seq_id + 1
                struct = __destructure(n, level, seq_id, struct)
        elif t.name == 'li':
            for n in t.children:
                struct = __destructure(n, level, seq_id, struct)
        elif t.name == 'a':
            ft = t.string
            key = len(struct)
            struct[key] = {
                'title': ft,
                'level': level,
                'seq_id': seq_id
            }
        else:
            pass

        return struct


    assert (isinstance(pdf_fp, str))
    if not pdf_fp:
        raise ValueError("structure_from_bs4outline - pdf_fp not provided")
    if not os.path.isfile(pdf_fp):
        raise ValueError("structure_from_bs4outline - file point at by pdf_fp does not exist")


    # we have to start at -1 to account for the zeroth ul
    bs4_html = outline_to_im(pdf_fp=pdf_fp, verbose=verbose)
    final_struct = __destructure(bs4_html.contents[0], -1, 0, {})

    return final_struct


def pdf_to_pagewise_html(pdf_fp,
                         pdf_title='',
                         pdf_author='',
                         doc_proc_fn  = (lambda page_dict, **kwargs: None),
                         page_proc_fn = (lambda fp, **kwargs: None),
                         **kwargs):
    """
    Take a pdf and process it into a representation that can be used to generate the intermediate form. For that,
    we split the processing into three steps.
    1. process the pdf into page-wise html documents using pdf2htmlex
    2. take the user function page_proc_fn (page_html_filepath to output) and run it over individual pages.
    3. take the processed individual pages struct and run this through the doc_proc_fn (page_out_dictionary -> document_dictionary)
    :param pdf_fp: (str) full path to pdf document
    :param doc_proc_fn:
    :param page_proc_fn:
    :param **kwargs:
    :return: the processed document
    """
    assert (isinstance(pdf_fp, str))
    assert (isinstance(doc_proc_fn, types.FunctionType))
    assert (isinstance(page_proc_fn, types.FunctionType))

    if not pdf_fp:
        raise ValueError('pdf_to_pagewise_html - pdf file path not provided')
    if not os.path.isfile(pdf_fp):
        raise ValueError('pdf_to_pagewise_html - pdf file does not exist')

    info = pdf.pdf_info(pdf_fp)

    verbose = kwargs.get('verbose', False)

    temp_dir = tempfile.gettempdir()
    temp_dir_fp = os.path.join(temp_dir, str(uuid.uuid1()))
    os.mkdir(temp_dir_fp)

    ret_success  = pdf2html(fp_pdf=pdf_fp,
                            generated_html_filename="generated.html",
                            generated_pagehtml_filename="generated_%d.html",
                            fp_outdir=temp_dir_fp,
                            splitPages=True,
                            optimizeText=False,
                            processNonText=True,
                            preprocessToPS=False,
                            options={
                                "process-outline": "0"
                            },
                            debug=verbose)

    if not ret_success:
        raise ValueError('pdf_to_pagewise_html - failed to generate html from pdf')

    temp_pattern = "%s%sgenerated_*.html" % (temp_dir_fp, os.sep)
    pages = {}
    for i, j_file_fp in enumerate(glob.glob(temp_pattern)):
        page_ret = page_proc_fn(j_file_fp, pdf_info=info, verbose=verbose)
        o = {
            'page_no': i,
            'path' : j_file_fp,
            'content': page_ret
        }
        pages[i] = o

    # remove the raw file content - not needed at this point
    keep_im = kwargs.get('keep_temp_files', False)
    if not keep_im:
        if verbose:
            print("Removing temporary files ...")
        shutil.rmtree(temp_dir_fp)

    # read back the generated html files - and return them
    out_struct = {'pdf_fp': pdf_fp,
                  'info'  : info,
                  'page_struct': pages}

    doc_proc = doc_proc_fn(out_struct, pdf_info=info, verbose=verbose)

    out_struct['doc_struct'] = doc_proc

    return out_struct