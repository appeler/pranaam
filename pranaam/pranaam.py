"""Entry point module for pranaam CLI."""

import argparse
import sys

from .logging import get_logger
from .naam import Naam

logger = get_logger()

# Export main prediction function
pred_rel = Naam.pred_rel


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point for religion prediction.

    Args:
        argv: Command line arguments, defaults to sys.argv[1:]

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Predict religion based on name",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, help="Name to analyze (single name as string)"
    )
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
        # Re-raise SystemExit for help and argument errors
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
