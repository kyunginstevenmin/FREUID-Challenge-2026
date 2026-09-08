"""Shared pytest setup: make src/ importable from any test without per-file boilerplate."""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
