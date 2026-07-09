"""Small cross-version shims for the Hugging Face stack.

pairjudge pins ``transformers>=4.46`` without an upper bound, so a fresh
install may resolve either the 4.x or the 5.x line. These helpers keep the
call sites working (and quiet) on both.
"""

from __future__ import annotations

from typing import Any


def model_dtype_kwargs(dtype: Any) -> dict[str, Any]:
    """Keyword for ``from_pretrained``'s weight dtype.

    transformers 5.0 renamed ``torch_dtype`` -> ``dtype``; passing the old
    name warns on 5.x and passing the new name errors on 4.x. Pick whichever
    the installed version expects.
    """
    from transformers import __version__ as tv

    try:
        major = int(tv.split(".")[0])
    except (ValueError, IndexError):
        major = 4
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}
