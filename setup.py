# Copyright (c) Facebook, Inc. and its affiliates.

from setuptools import find_packages, setup

setup(
    name='dcem',
    version='0.0.1',
    description="The Differentiable Cross-Entropy Method",
    author='Brandon Amos',
    author_email='brandon.amos.cs@gmail.com',
    platforms=['any'],
    license="CC BY-NC 4.0",
    url='https://github.com/facebookresearch/dcem',
    py_modules=['dcem'],
    install_requires=[
        'numpy>=1<2',
        'higher',
        'setproctitle',
        'lml@git+git://github.com/locuslab/lml.git',
    ],
    dependency_links=[
        'git+ssh://git@github.com/locuslab/lml.git#egg=lml-0.0.1'
        'git+git://github.com/denisyarats/dmc2gym.git'
    ]
)
