#!/usr/bin/env python3
import logging
import argparse


def main():
    logging.debug("Hey, I'm a debug message!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
