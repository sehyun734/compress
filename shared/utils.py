from dataclasses import fields
from typing import Any


def print_args(args: Any, skip: list[str] = ["model"]) -> None:
    for f in fields(args):
        if f.name not in skip:
            print(f"{f.name}={getattr(args, f.name)!r}")
    print()
