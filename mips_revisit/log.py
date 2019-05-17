"""
This code used to be Michael Whittaker's, but I've adapted it since.

Imagine you want to log stuff in multiple files of a Python project. You'd
probably write some code that looks like this:

    logging.debug("Hello, World!")

Pretty simple. Unfortunately, some packages like gym seem to clobber the
default logger that is used by logging.debug. No big deal, just use a different
logger:

    logging.getLogger("mylogger").debug("Hello, World!")

Two problems stick out though. First, "mylogger" is a bit of a magic constant.
Second, that's a lot of typing for a print statement! This file makes it a
tinier bit more convenient to log stuff. In one file, run this:

    import log
    log.init(verbose=True)

And then from any other file, run something this:

    from log import debug
    debug("Iteration {} of {}", 1, num_iters)
"""

import inspect
import logging
import os
from datetime import datetime

from absl import flags

flags.DEFINE_boolean("verbose", True, "whether to activate debug logging")


class _StackCrawlingFormatter(logging.Formatter):
    """
    If we configure a python logger with the format string
    "%(pathname):%(lineno): %(message)", messages logged via `log.debug` will
    be prefixed with the path name and line number of the code that called
    `log.debug`. Unfortunately, when a `log.debug` call is wrapped in a helper
    function (e.g. debug below), the path name and line number is always that
    of the helper function, not the function which called the helper function.

    A _StackCrawlingFormatter is a hack to log a different pathname and line
    number. Simply set the `pathname` and `lineno` attributes of the formatter
    before you call `log.debug`. See `debug` below for an example.
    """

    def __init__(self, format_str):
        super().__init__(format_str)
        self.pathname = None
        self.lineno = None

    def format(self, record):
        s = super().format(record)
        if self.pathname is not None:
            s = s.replace("{pathname}", self.pathname)
        if self.lineno is not None:
            s = s.replace("{lineno}", str(self.lineno))
        if "{fmttime}" in s:
            tztime = datetime.now().astimezone()
            fmttime = tztime.strftime("%Y-%m-%d %H:%M:%S %Z")
            s = s.replace("{fmttime}", fmttime)
        return s


_LOGGER = logging.getLogger(__package__)
_FORMAT_STRING = "[{fmttime} {pathname}:{lineno}] %(message)s"
_FORMATTER = _StackCrawlingFormatter(_FORMAT_STRING)


def init(verbose=None):
    """Initialize the logger (uses flag value by default)"""
    handler = logging.StreamHandler()
    handler.setFormatter(_FORMATTER)
    _LOGGER.propagate = False
    _LOGGER.addHandler(handler)
    if verbose is None:
        verbose = flags.FLAGS.verbose
    if verbose:
        _LOGGER.setLevel(logging.DEBUG)
    else:
        _LOGGER.setLevel(logging.INFO)


def _prep_formatter():
    # Get the path name and line number of the function which called us.
    previous_frame = inspect.currentframe().f_back.f_back
    try:
        pathname, lineno, _, _, _ = inspect.getframeinfo(previous_frame)
    except Exception:  # pylint: disable=broad-except
        pathname = "<UNKNOWN-FILE>.py"
        lineno = 0
    _FORMATTER.pathname = _clean_path(pathname)
    _FORMATTER.lineno = lineno


def debug(s, *args):
    """debug(s, x1, ..., xn) logs s.format(x1, ..., xn)."""
    _prep_formatter()
    _LOGGER.debug(s.format(*args))


def info(s, *args):
    """info(s, x1, ..., xn) logs s.format(x1, ..., xn)."""
    _prep_formatter()
    _LOGGER.info(s.format(*args))


def warn(s, *args):
    """warn(s, x1, ..., xn) logs s.format(x1, ..., xn)."""
    _prep_formatter()
    _LOGGER.warn(s.format(*args))


def error(s, *args):
    """error(s, x1, ..., xn) logs s.format(x1, ..., xn)."""
    _prep_formatter()
    _LOGGER.error(s.format(*args))


def fatal(s, *args):
    """fatal(s, x1, ..., xn) logs s.format(x1, ..., xn)."""
    _prep_formatter()
    _LOGGER.fatal(s.format(*args))


def _clean_path(path):
    """
    Simplifies the path for readability.
    """
    path = os.path.abspath(path)
    home = os.path.expanduser("~")
    if os.path.commonpath([path, home]) == home:
        home_path = os.path.join("~", os.path.relpath(path, home))
        home_path = os.path.normpath(home_path)
    else:
        home_path = path
    cwd = os.getcwd()
    if os.path.commonpath([cwd, path]):
        cwd_path = os.path.join(".", os.path.relpath(path, cwd))
        cwd_path = os.path.normpath(cwd_path)
    else:
        cwd_path = path
    return min(home_path, cwd_path, path, key=len)
