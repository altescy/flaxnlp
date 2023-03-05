import argparse
import json

import colt

from flaxnlp.commands.subcommand import Subcommand
from flaxnlp.tasks import Task


@Subcommand.register("run")
class RunCommand(Subcommand):
    def setup(self) -> None:
        self.parser.add_argument("config")
        self.parser.add_argument("output")
        self.parser.add_argument("--include-package", action="append", default=[])

    def run(self, args: argparse.Namespace) -> None:
        colt.import_modules(args.include_package)

        with open(args.config) as jsonfile:
            config = json.load(jsonfile)

        task = colt.build(config, Task)
        task.run(args.output)
