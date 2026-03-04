"""Voice2Text package."""

import os

# Must be set before huggingface_hub is imported anywhere.
# Both hf_xet and tqdm spawn subprocesses (xet for downloads, tqdm for a
# multiprocessing RLock via its resource tracker).  Inside Textual's worker
# threads the fd table is in a bad state, so Python 3.14's strict
# subprocess fd validation crashes with "bad value(s) in fds_to_keep".
# We have our own progress callback, so neither is needed.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
