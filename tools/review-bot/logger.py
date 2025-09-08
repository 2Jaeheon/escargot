import logging
import os
import sys


def get_logger(name: str = "review-bot") -> logging.Logger:
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger

	# Level from environment (default DEBUG)
	logger.setLevel(os.getenv("LOG_LEVEL", "DEBUG").upper())

	# Always log to stdout with minimal unified format
	handler = logging.StreamHandler(sys.stdout)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	# Avoid duplicate logs via root
	logger.propagate = False
	return logger
