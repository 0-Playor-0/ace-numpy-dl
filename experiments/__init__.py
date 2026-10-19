"""Experiment scripts. Every script accepts --smoke (anywhere on the command line): it sets ACE_SMOKE=1
before acenet is imported, which shrinks the data to fixed tiny subsets and every training run to one epoch
(acenet.data.SMOKE). Smoke runs only check that the pipeline executes; scripts/smoke_test.py runs them all in a
fresh clone so the committed runs/ are never touched."""

import os
import sys

if "--smoke" in sys.argv:
    sys.argv.remove("--smoke")
    os.environ["ACE_SMOKE"] = "1"
