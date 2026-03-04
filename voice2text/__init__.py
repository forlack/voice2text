"""Voice2Text package."""

import os

# Must be set before huggingface_hub is imported anywhere.
# hf_xet spawns subprocesses that trigger Python 3.14 FDS_TO_KEEP errors.
# Disabling xet forces huggingface_hub to use plain HTTP downloads instead.
os.environ["HF_HUB_DISABLE_XET"] = "1"
