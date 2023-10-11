import logging
import os

__version__ = "0.0.31"

logging.basicConfig(level=logging.INFO)

# Configurable variables

# Whether to skip column type checking or not
CHECK_COL_TYPES: bool = True
# If type check, how many rows to fetch in order to check
FETCH_ROW_NUM: int = 10
# For all functions that expect binary target, should we do a safeguard?

# Number of threads to use by default in non-Polars settings.
THREADS:int = os.cpu_count() - 1
# Whether to persis in Blueprint or not
PERSIST_IN_BLUEPRINT = True
# Whether to stream internal collect or not
STREAM_TRANSFORM = False # Whether to stream when collecting in dsds.transform
