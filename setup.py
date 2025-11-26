"""
Minimal setup.py for backward compatibility.

Modern Python packaging uses pyproject.toml (PEP 517/518).
This file exists only for compatibility with older build tools.

For package configuration, see pyproject.toml.
"""

from setuptools import setup

# All configuration is now in pyproject.toml
# This setup() call is intentionally minimal for backward compatibility
setup()