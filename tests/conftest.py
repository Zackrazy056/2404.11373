import os
from pathlib import Path


_TEST_HOME = Path(__file__).resolve().parent / ".tmp_home"
_TEST_HOME.mkdir(parents=True, exist_ok=True)
(_TEST_HOME / "arviz_data").mkdir(parents=True, exist_ok=True)

os.environ["HOME"] = str(_TEST_HOME)
os.environ["USERPROFILE"] = str(_TEST_HOME)
os.environ["ARVIZ_DATA"] = str(_TEST_HOME / "arviz_data")
