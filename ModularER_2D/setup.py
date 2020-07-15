#!/usr/bin/env python
from setuptools import setup

setup(name='LSystem',
      description="Modular robotics environment for OpenAi-Gym",
      version='0.1.0',
      keywords="modular robotics gym openai-gym",
      author="Jørgen Nordmoen and Frank Veenstra",
      author_email="jorgehn@ifi.uio.no",
      include_package_data=True,
      install_requires=['gym>=0.15', 'pybullet>=2.5', 'numpy>=1.17'],
      test_suite='tests')
