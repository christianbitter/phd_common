import io
import os
import tempfile
import uuid


def get_temp_file(suffix=None):
    """
    generate the file path to a temporary file.
    :param suffix: (str) optional suffix/ file extension to provide
    :return: the generated temporary file path
    """
    temp_file_name = str(uuid.uuid4())
    if suffix:
        temp_file_name = "%s%s" % (temp_file_name, suffix)

    tmp_fp = os.path.join(tempfile.gettempdir(), temp_file_name)

    return tmp_fp


def generate_dummy_html(html_fp, content, verbose=False):
    assert (isinstance(html_fp, str))

    if not html_fp:
        raise ValueError('generate_dummy_html - html file path not provided')

    if verbose:
        print("Writing: %s" % html_fp)

    with io.open(file=html_fp, mode='w+', encoding='utf-8', closefd=True) as im_file:
        im_file.writelines("<html><head><title></title></head><body>".decode("utf-8"))
        if content:
            im_file.writelines(content + "<br/>".decode("utf-8"))
        im_file.writelines("</body></html>".decode("utf-8"))