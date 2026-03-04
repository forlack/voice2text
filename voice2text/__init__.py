"""Voice2Text package."""

import os

# Must be set before huggingface_hub is imported anywhere.
# hf-xet spawns subprocesses that trigger Python 3.14 FDS_TO_KEEP errors.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
