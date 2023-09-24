import logging
import os


__version__ = "0.0.3"

logging.basicConfig(level=logging.INFO)

# Configurable variables

CHECK_COL_TYPES: bool = True

# Number of threads to use by default in non-Polars settings.
THREADS:int = os.cpu_count() - 1
# Whether to persis in Blueprint or not
PERSIST_IN_BLUEPRINT = True
# Whether to stream internal collect or not
STREAM_TRANSFORM = False # Whether to stream when collecting in dsds.transform

