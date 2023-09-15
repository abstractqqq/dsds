import logging
import os


__version__ = "0.0.26"

logging.basicConfig(level=logging.INFO)

# Configurable variables

CHECK_COL_TYPES: bool = True

# Not Polars, number of threads to use by default.
THREADS:int = os.cpu_count() - 1
