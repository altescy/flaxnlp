import argparse
from typing import Optional

from flaxnlp import __version__
from flaxnlp.commands import run  # noqa: F401
from flaxnlp.commands.subcommand import Subcommand


def create_subcommand(prog: Optional[str] = None) -> Subcommand:
    parser = argparse.ArgumentParser(usage="%(prog)s", prog=prog)
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__,
    )
    return Subcommand(parser)


def main(prog: Optional[str] = None) -> None:
    app = create_subcommand(prog)
    app()
