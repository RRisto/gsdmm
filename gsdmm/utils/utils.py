import sys
from contextlib import contextmanager
from six import string_types, iteritems
from smart_open import open

if sys.version_info[0] >= 3:
    unicode = str

"""pieces taken from gensim.utils.py"""


def file_or_filename(input):
    """Open a filename for reading with `smart_open`, or seek to the beginning if `input` is an already open file.

    Parameters
    ----------
    input : str or file-like
        Filename or file-like object.

    Returns
    -------
    file-like object
        An open file, positioned at the beginning.

    """
    if isinstance(input, string_types):
        # input was a filename: open as file
        return open(input, 'rb')
    else:
        # input already a file-like object; just reset to the beginning
        input.seek(0)
        return input


@contextmanager
def open_file(input):
    """Provide "with-like" behaviour without closing the file object.

    Parameters
    ----------
    input : str or file-like
        Filename or file-like object.

    Yields
    -------
    file
        File-like object based on input (or input if this already file-like).

    """
    mgr = file_or_filename(input)
    exc = False
    try:
        yield mgr
    except Exception:
        # Handling any unhandled exceptions from the code nested in 'with' statement.
        exc = True
        if not isinstance(input, string_types) or not mgr.__exit__(*sys.exc_info()):
            raise
        # Try to introspect and silence errors.
    finally:
        if not exc and isinstance(input, string_types):
            mgr.__exit__(None, None, None)


def revdict(d):
    """Reverse a dictionary mapping, i.e. `{1: 2, 3: 4}` -> `{2: 1, 4: 3}`.

    Parameters
    ----------
    d : dict
        Input dictionary.

    Returns
    -------
    dict
        Reversed dictionary mapping.

    Notes
    -----
    When two keys map to the same value, only one of them will be kept in the result (which one is kept is arbitrary).

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.utils import revdict
        >>> d = {1: 2, 3: 4}
        >>> revdict(d)
        {2: 1, 4: 3}

    """
    return {v: k for (k, v) in iteritems(dict(d))}


def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a unicode or bytes string in the given encoding into a utf8 bytestring.

    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour if `text` is a bytestring.
    encoding : str, optional
        Encoding of `text` if it is a bytestring.

    Returns
    -------
    str
        Bytestring in utf8.

    """

    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


to_utf8 = any2utf8


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert `text` (bytestring in given encoding or unicode) to unicode.
    Parameters
    ----------
    text : str
        Input text.
    errors : str, optional
        Error handling behaviour if `text` is a bytestring.
    encoding : str, optional
        Encoding of `text` if it is a bytestring.
    Returns
    -------
    str
        Unicode version of `text`.
    """
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)


to_unicode = any2unicode
