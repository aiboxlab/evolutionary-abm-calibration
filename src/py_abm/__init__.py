"""Arquivo de inicialização.
"""
import logging.config
import sys

LOGGING = {
    'version': 1,
    'formatters': {
        'verbose': {
            'format': ('[{asctime}: {levelname}] [{filename}] [PID: {process:d}] '
                       '[Thread ID: {thread:d}] {message}'),
            'style': '{',
        },
        'brief': {
            'format': '[{asctime}: {levelname}] [{filename}] {message}',
            'style': '{',
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'brief',
            'stream': sys.stdout
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'formatter': 'verbose',
            'filename': 'logs.log',
            'mode': 'a'
        }
    },
    'loggers': {
        'py_abm': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        }
    }
}

logging.config.dictConfig(LOGGING)
