#!/usr/bin/env python3

import sys
import os

# Add parsit to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parsit.train.train import train

if __name__ == "__main__":
    train()