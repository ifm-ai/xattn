"""Run with python -m tests.softdelta.benchmark."""
from tests._benchmark import main

if __name__ == '__main__':
    main(('softdelta_full', 'softdelta_window', 'softdelta_chunk'))
