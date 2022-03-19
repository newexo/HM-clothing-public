import os


def qualifyname(directoryname, filename=None):
    if filename is None:
        return directoryname
    return os.path.join(directoryname, filename)


def code(filename=None):
    codepath = os.path.dirname(__file__)
    return qualifyname(codepath, filename)


def base(filename=None):
    basepath = os.path.abspath(code(".."))
    return qualifyname(basepath, filename)


def tests(filename=None):
    return qualifyname(os.path.join(code(), "tests"), filename)


def data(filename=None):
    return qualifyname(os.path.join(base(), "data"), filename)


def testdata(filename=None):
    return qualifyname(os.path.join(tests(), "testdata"), filename)
