import os
import re
import codecs


def __is_content_block(b, to_ascii=True):
    if not b:
        return False

    if to_ascii:
        b = b.encode('ascii', 'ignore')

    rx_block = "\<\w:\w+\>"
    m = re.match(pattern=rx_block, string=b)
    return m is not None


def __get_content_block(b, to_ascii=True):
    if to_ascii:
        b = b.encode('ascii', 'ignore')

    rx_block = "\<(?P<Mode>\w):(?P<Type>\w+)\>"
    m = re.match(pattern=rx_block, string=b)
    cb_mode = m.group('Mode')
    cb_type = m.group('Type')

    return cb_mode.lower(), cb_type.lower()


def intermediate_csv_to_content_block(csv_fp, encoding, to_ascii = False, verbose=False):
    assert (isinstance(csv_fp, str))

    if not os.path.isfile(csv_fp):
        raise ValueError("intermediate_csv_to_content_block - csv_fp does not point to file")

    content_blocks = []
    block_stack = []

    with codecs.open(csv_fp,'r', encoding=encoding) as f:
        active_block = None
        content = ""
        for line in f:
            line = line.strip()

            if not line:
                continue

            if verbose:
                print("Processing: %s" % line)

            if __is_content_block(line):
                cb_mode, cb_type = __get_content_block(line)

                # two cases for content moving ...
                # we close an existing tag -> assign the content to the closed tag
                # we open a new tag and we have content -> assign it to the opened tag
                if cb_mode == "b":
                    block_stack.append(cb_type)
                    if content and cb_type != active_block:
                        b = {active_block: content}
                        content_blocks.append(b)
                        content = ""

                    active_block = cb_type
                else:
                    # if we try to close a type that is not open raise error
                    last_type = block_stack.pop()

                    if last_type != cb_type:
                        if verbose:
                            print("Block-Stack: %s" % block_stack)

                        raise ValueError('Stack (last vs closing) - %s vs. %s' % (last_type, cb_type))

                    if content:
                        b = {cb_type: content}
                        content_blocks.append(b)

                    content = ""
                    active_block = None
            else:
                if active_block is None:
                    raise ValueError("""intermediate_csv_to_content_block - 
                                        cannot associate content line to None content block.\r\nSee: %s""" % line)
                else:
                    if to_ascii:
                        line = line.encode('ascii', 'ignore')

                    content = content + " " + line.strip()
                    content = content.strip()

    if block_stack:
        raise ValueError("intermediate_csv_to_content_block - not all types processed\r\n"
                         "It seems that the intermediate file is structurally not intact.")

    return content_blocks
