#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

dist = setup(
    name="mmdagg",
    version="1.0.0",
    description="Maximum Mean Discrepancy Aggregated Test",
    author="Antonin Schrab",
    author_email="a.lastname@ucl.ac.uk",
    license="MIT License",
    packages=["mmdagg", ],
    install_requires=["numpy", "scipy", "jax", "jaxlib"],
    python_requires=">=3.9",
)
