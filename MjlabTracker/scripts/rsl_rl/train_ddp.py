"""Distributed training entry point.

mjlab uses the same native training CLI; launch it under your torchrun/deepspeed
wrapper if distributed execution is needed.
"""

from train import main


if __name__ == "__main__":
    main()
