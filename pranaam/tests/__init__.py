import sys
from contextlib import contextmanager
from typing import Generator, Any, Callable
from io import StringIO


@contextmanager
def capture(command: Callable[..., Any], *args: Any, **kwargs: Any) -> Generator[str, None, None]:
    out, sys.stdout = sys.stdout, StringIO()
    command(*args, **kwargs)
    sys.stdout.seek(0)
    yield sys.stdout.read()
    sys.stdout = out
