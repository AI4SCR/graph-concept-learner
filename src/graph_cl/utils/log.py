import functools
import logging
import sys
from datetime import datetime
from pathlib import Path

# ANSI escape codes for various colors
COLORS = {
    "HEADER": "\033[95m",
    "OKBLUE": "\033[94m",
    "OKCYAN": "\033[96m",
    "OKGREEN": "\033[92m",
    "WARNING": "\033[93m",
    "FAIL": "\033[91m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m",
    "UNDERLINE": "\033[4m",
}

LEVEL_COLORS = {
    logging.DEBUG: COLORS["OKBLUE"],
    logging.INFO: COLORS["OKGREEN"],
    logging.WARNING: COLORS["WARNING"],
    logging.ERROR: COLORS["FAIL"],
    logging.CRITICAL: COLORS["FAIL"],
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        if levelname in LEVEL_COLORS:
            levelname_color = LEVEL_COLORS[record.levelno] + levelname + COLORS["ENDC"]
            record.levelname = levelname_color
        return super().format(record)


# create logger
logger = logging.getLogger("gcl")
# note: this determines with messages are passed on to the handlers, thus we keep it at DEBUG
logger.setLevel(logging.DEBUG)


# create stream handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.name = "stdout"
console_handler.setLevel(logging.DEBUG)  # Adjust as needed
console_formatter = ColorFormatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

log_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"{log_id}.log"

# logger cache folder
log_dir = Path.home() / ".logging" / logger.name
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / log_filename

file_handler = logging.FileHandler(log_path)
file_handler.name = "file"
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


def set_stdout_stream_level(verbose: int = 0, level: None | int = None):
    if level:
        logger.setLevel(level)
        return

    if verbose == 0:
        logging_level = logging.ERROR
    elif verbose == 1:
        logging_level = logging.WARNING
    elif verbose == 2:
        logging_level = logging.INFO
    else:
        logging_level = logging.DEBUG

    console_handler.setLevel(logging_level)


def enable_verbose(func):
    @functools.wraps(func)
    def function_with_verbose(*args, verbose=1, **kwargs):
        level = logging.getLogger().level
        set_stdout_stream_level(verbose)
        result = func(*args, **kwargs)
        set_stdout_stream_level(level)
        return result

    return function_with_verbose
