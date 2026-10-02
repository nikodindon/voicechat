"""Rend le paquet `voicechat` importable depuis tests/ sans installation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
