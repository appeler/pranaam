"""Entry point for Pranaam name-pattern estimation."""

import argparse
import sys

from .logging import get_logger
from .naam import Naam

logger = get_logger()

pred_rel = Naam.pred_rel


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
        description="Estimate patterns associated with a name",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Name to analyze")
    parser.add_argument(
        "--lang", default="eng", choices=["eng", "hin"], help="Language of input name"
    )
    parser.add_argument(
        "--latest", action="store_true", help="Download latest model version"
    )

    try:
        args = parser.parse_args(argv)
        result = pred_rel(args.input, lang=args.lang, latest=args.latest)
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
