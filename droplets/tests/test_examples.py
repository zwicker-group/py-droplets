"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import glob
import os
import subprocess as sp
import sys
from pathlib import Path
from typing import List  # @UnusedImport

import pytest

from pde.tools.misc import module_available

PACKAGE_PATH = Path(__file__).resolve().parents[2]
EXAMPLES = glob.glob(str(PACKAGE_PATH / "examples" / "*.py"))

SKIP_EXAMPLES: List[str] = []
if not module_available("matplotlib"):
    SKIP_EXAMPLES.append("plot_emulsion.py")


@pytest.mark.no_cover
@pytest.mark.skipif(sys.platform == "win32", reason="Assumes unix setup")
@pytest.mark.parametrize("path", EXAMPLES)
def test_example(path):
    """runs an example script given by path"""
    if os.path.basename(path).startswith("_"):
        pytest.skip("skip examples starting with an underscore")
    if any(name in path for name in SKIP_EXAMPLES):
        pytest.skip(f"Skip test {path}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_PATH) + ":" + env.get("PYTHONPATH", "")
    proc = sp.Popen([sys.executable, path], env=env, stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        outs, errs = proc.communicate(timeout=30)
    except sp.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()

    msg = "Script `%s` failed with following output:" % path
    if outs:
        msg = "%s\nSTDOUT:\n%s" % (msg, outs)
    if errs:
        msg = "%s\nSTDERR:\n%s" % (msg, errs)
    assert proc.returncode <= 0, msg
