"""Entry point for Pranaam name-pattern estimation."""

import argparse
import sys

from .logging import get_logger
from .naam import Naam

logger = get_logger()

estimate_muslim_name_pattern = Naam.estimate_muslim_name_pattern


def main(argv: list[str] | None = None) -> int:
    """Run name-pattern estimation from the command line.

    Args:
        argv: Command line arguments, defaults to sys.argv[1:]

    Returns:
        int: Exit code (0 for success, non-zero for error)

    Raises:
        SystemExit: For help and argument parsing errors
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Estimate Muslim-associated patterns in a name",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Name to analyze")
    parser.add_argument(
        "--lang", default="eng", choices=["eng", "hin"], help="Language of input name"
    )
    parser.add_argument(
        "--refresh-pinned",
        action="store_true",
        help="Reload and verify the immutable pinned model artifacts",
    )

    try:
        args = parser.parse_args(argv)
        result = estimate_muslim_name_pattern(
            args.input,
            lang=args.lang,
            refresh_pinned=args.refresh_pinned,
        )
        print(result.to_string(index=False))
        return 0

    except SystemExit:
        raise
    except Exception as e:
        error_message = f"Error: {e}"
        logger.error(error_message)
        print(error_message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
