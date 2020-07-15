#!/usr/bin/env python

"""
Exception module for morphologies
"""


class MorphologyError(Exception):
    """Base exception for all morphological errors"""
    pass


class NoModuleAttached(MorphologyError):
    """No module was attached at the key"""
    pass


class ModuleAttached(MorphologyError):
    """A module is already connected at the desired key"""
    pass


class NoAvailable(MorphologyError):
    """There were no available connection points"""
    pass


class ConnectionObstructed(MorphologyError):
    """The desired connection point was obstructed and the module could not be
    attached"""
    pass
