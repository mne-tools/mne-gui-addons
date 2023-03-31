# -*- coding: utf-8 -*-
"""The documentation functions."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from mne.utils.docs import _indentcount_lines, docdict as _mne_docdict

##############################################################################
# Define our standard documentation entries

docdict = _mne_docdict.copy()
# Specific to this repo

docdict[
    "tmax"
] = """
tmax : scalar
    Time point of the last sample in data.
"""

docdict_indented = {}


def _fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.
    """
    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = " " * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = "\n".join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError(
            "%s documenting %s:\n%s" % (exp.__class__.__name__, funcname, str(exp))
        )
    return f
