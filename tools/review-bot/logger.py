import logging
import logging.handlers
import os

try:
	from systemd.journal import JournalHandler  # type: ignore
	_HAS_JOURNAL = True
except Exception:
	_HAS_JOURNAL = False


def get_logger(name: str = "review-bot") -> logging.Logger:
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger

	logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

	if _HAS_JOURNAL:
		handler = JournalHandler(SYSLOG_IDENTIFIER=name)
	else:
		handler = logging.StreamHandler()
		fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
		handler.setFormatter(logging.Formatter(fmt))

	logger.addHandler(handler)
	logger.propagate = False
	return logger
