import tidyrun
import re


def test_version():
    """Version must match [N!]N(.N)*[{a|b|rc}N][.postN][.devN], cf. PEP 440"""
    assert isinstance(tidyrun.__version__, str)
    pattern = r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$"
    assert re.match(pattern, tidyrun.__version__)
