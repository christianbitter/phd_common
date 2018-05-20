import io


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