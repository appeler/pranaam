import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from io import StringIO
from typing import Any


@contextmanager
def capture(
    command: Callable[..., Any], *args: Any, **kwargs: Any
) -> Generator[str, None, None]:
    out, sys.stdout = sys.stdout, StringIO()
    command(*args, **kwargs)
    sys.stdout.seek(0)
    yield sys.stdout.read()
    sys.stdout = out
