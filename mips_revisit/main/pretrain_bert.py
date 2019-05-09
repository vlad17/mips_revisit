"""
Usage: python schedule.py < input

See README.md for input format
"""

from absl import app, flags

from .. import log

flags.DEFINE_enum(
    "attention",
    "approx-mips",
    ["exact-mips", "approx-mips", "soft"],
    "type of attention method to use during training",
)


def _main(_argv):
    log.init()
    K = 10

    log.info("Hello, World! K = {}", K)


if __name__ == "__main__":
    app.run(_main)
