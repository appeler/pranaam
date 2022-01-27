import sys
import argparse

from .naam import Naam

pred_rel = Naam.pred_rel


def main(argv=sys.argv[1:]):
    title = "Predict religion based on name"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--input", default=None, help="name")
    args = parser.parse_args(argv)
    print(args)
    if not args.input or not args.type:
        return -1

    output = pred_rel(args.input)
    print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
