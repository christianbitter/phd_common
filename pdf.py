import os
import io
import tempfile
from subprocess import call


class PDFInfo(object):
    """
    Wrapper object for the xpdftools pdfinfo command
    """
    def __init__(self, title, subject, keywords, author, producer, creation_date, modified_date, is_tagged, form,
                 no_pages, is_encrypted, page_size, file_size, is_optimized, version):
        self.title = title
        self.subject = subject
        self.keywords = keywords
        self.author = author
        self.producer = producer
        self.creation_date = creation_date
        self.modified_date = modified_date
        self.tagged = is_tagged
        self.form = form
        self.no_pages = int(no_pages)
        self.encrypted = is_encrypted
        self.page_size = page_size
        self.file_size = file_size
        self.optimized = is_optimized
        self.version = version

    @property
    def vKeywords(self):
        sep = ","
        if self.keywords.find(",") < 0:
            sep = ";"
        return [kw.strip() for kw in self.keywords.split(sep)]

    @property
    def isTagged(self):
        return self.tagged.lower() != "no"

    @property
    def isOptimized(self):
        return self.optimized.lower() != "no"

    @property
    def isEncrypted(self):
        return self.encrypted.lower() != "no"

    def __repr__(self):
        return  ("Title: %s\r\nSubject: %s\r\nKeywords: %s\r\nAuthor: %s\r\nProducer: %s\r\nCreation Date: %s\r\n" + \
                 "Modification Date: %s\r\nIs Tagged: %s\r\nForm: %s\r\nNumber of Pages: %s\r\nIs Encrypted: %s\r\n" + \
                 "Page Size: %s\r\nFile Size: %s\r\nIs Optimized: %s\r\nPDF Version: %s") % (self.title,
                                                                                             self.subject,
                                                                                             self.keywords,
                                                                                             self.author,
                                                                                             self.producer,
                                                                                             self.creation_date,
                                                                                             self.modified_date,
                                                                                             self.isTagged,
                                                                                             self.form,
                                                                                             self.no_pages,
                                                                                             self.isEncrypted,
                                                                                             self.page_size,
                                                                                             self.file_size,
                                                                                             self.isOptimized,
                                                                                             self.version)

def pdf_info(pdf_fp,
             pdfinfo="C:/Development/xpdf-tools-win-4.00/bin64/pdfinfo.exe"):
    """
    uses xpdftools' pdfinfo module to get the info section of a provided pdf file
    :param pdf_fp: (str) path to pdf file
    :param pdfinfo: (str) path to xpdftools' pdfinfo executable
    :return: PDFInfo object
    """
    assert( isinstance(pdf_fp, str))
    if not pdf_fp:
        raise ValueError("pdf_info - pdf file not provided")
    if not os.path.isfile(pdf_fp):
        raise ValueError("pdf_info - path to pdf file not found")

    pdf2html_args = [pdfinfo, pdf_fp]
    fp_dir_pdfinfo = os.path.dirname(pdfinfo)

    pdf_info_tmp = os.path.join(tempfile.gettempdir(), tempfile.gettempprefix())

    retcode = call(args=pdf2html_args,
                   shell=False,
                   stdout=open(pdf_info_tmp, 'w+'),
                   cwd=fp_dir_pdfinfo)

    if retcode == 0:
        out_struct = {
            'Title': '',
            'Subject': '',
            'Keywords': '',
            'Author': '',
            'Creator': '',
            'CreationDate': '',
            'ModDate': '',
            'Tagged': '',
            'Form': '',
            'Pages': '',
            'Encrypted': '',
            'Page size': '',
            'File size': '',
            'Optimized': '',
            'PDF version': ''
        }

        with io.open(file=pdf_info_tmp, mode='r+') as info_file:
            for aline in info_file.readlines():
                for a_key in out_struct.keys():
                    key_match = "%s:" % a_key
                    if aline.startswith(key_match):
                        key_val = aline.replace(key_match, "").strip()
                        out_struct[a_key] = key_val
                        next

            pdf_info = PDFInfo(title=out_struct.get('Title', None),
                               subject=out_struct.get('Subject', None),
                               keywords=out_struct.get('Keywords', None),
                               author=out_struct.get('Author', None),
                               producer=out_struct.get('Producer', None),
                               creation_date=out_struct.get('CreationDate', None),
                               modified_date=out_struct.get('ModDate', None),
                               is_tagged=out_struct.get('Tagged', None),
                               form=out_struct.get('Form', None),
                               no_pages=out_struct.get('Pages', None),
                               is_encrypted=out_struct.get('Encrypted', None),
                               page_size=out_struct.get('Page size', None),
                               file_size=out_struct.get('File size', None),
                               is_optimized=out_struct.get('Optimized', None),
                               version=out_struct.get('PDF version', None))
            return pdf_info
    else:
        raise ValueError('pdf_info - Failed to pdfinfo on provided pdf file')