#!/usr/bin/env python

"""
Abstract module which all morphological modules must extend
"""
from collections import deque
import numpy as np


class Module(object):
    """Abstract morphological module"""
    # Possible connected parent module
    parent = None
    # Connection to parent, should be 'None' or a tuple of position, direction
    connection = None
    # Subclasses should point their respective connection classes here
    connection_type = None
    # Connection ID to use when creating constraint
    connection_id = -1
    # Dictionary of children, defined here to make abstract methods more powerful
    _children = None

    @property
    def children(self):
        """
        Get a list of the immediate children of this module.

        This is different from '__iter__' in that this method only returns the
        children directly connected to this module and not their children.
        """
        res = []
        if self.connection_type:
            for conn in self.connection_type:
                if conn in self._children:
                    res.append(self._children[conn])
        return res

    @property
    def available(self):
        """
        Get the list of available connection points that can be used to add new
        modules.

        This function should return a list of items corresponding to this
        modules access type.
        """
        res = []
        if self.connection_type:
            for conn in self.connection_type:
                if conn not in self._children:
                    res.append(conn)
        return res

    @property
    def joint(self):
        """
        Get the joint configuration as a dict.

        The joint configuration is a leaky abstraction used for movable joints.
        Subclasses should return a dict with joint configuration according to
        pybyllet, supported items are 'controlMode' (required), 'jointIndex'
        (required), 'positionGain' (optional), 'velocityGain' (optional) and
        'maxVelocity' (optional).

        Non-movable joints should return None.
        """
        return None

    @property
    def root(self):
        """Extract the root module of this module.

        This method will iterate through all parents until the root is found.
        If this module does not have a parent `self` will be returned."""
        queue = self
        while True:
            if queue.parent is None:
                return queue
            queue = queue.parent

    def connection_point(self, item):
        """Get the connection point associated with 'item'"""
        if not isinstance(item, (Module, self.connection_type)):
            raise TypeError("Cannot connect {} to modules".format(item))
        if item not in self:
            raise KeyError("This module has no child: '{}'".format(item))
        for conn, child in self._children.items():
            if child == item:
                return conn
        raise KeyError("This module has no child: '{}'".format(item))

    def __iter__(self):
        """Iterate all modules connected to this module"""
        yield self
        queue = deque([self])
        while queue:
            for child in queue.popleft().children:
                yield child
                queue.append(child)

    def __contains__(self, item):
        """Check if the module 'item' is connected to this module"""
        if isinstance(item, Module):
            return item in self._children.values()
        if self.connection_type and isinstance(item, self.connection_type):
            return item in self._children.keys()
        raise TypeError("Cannot connect {} to modules".format(item))

    def __len__(self):
        """
        Get the size of the morphology represented by this module.

        This method should count the length of all children also connected to
        this module.
        """
        return 1 + sum([len(m) for m in self.children])

    def __getitem__(self, key):
        """Return the child connected at 'key'"""
        if self.connection_type and not isinstance(key, self.connection_type):
            raise TypeError("Key: '{}' is not a Connection type".format(key))
        if key not in self:
            raise NoModuleAttached("No module attached at: {}".format(key))
        return self._children[key]

    def __delitem__(self, key):
        """Remove the connection to child connected at 'key'"""
        # Check that 'key' is correct type
        if not isinstance(key, (Module, self.connection_type)):
            raise TypeError("Key: '{}' is not a supported connection type"
                            .format(key))
        # If we are asked to delete a module from our self
        if isinstance(key, Module):
            key = self.connection_point(key)
        # Check if key is in children
        if key not in self:
            raise NoModuleAttached("No module attached at: {}".format(key))
        # First get reference to child so that we can update
        child = self._children[key]
        # Delete child from our children
        del self._children[key]
        # Update child so that its position and orientation is updated
        child.update()

    def __setitem__(self, key, module):
        """Attached the child 'module' at the point 'key'"""
        raise NotImplementedError("Not supported")

    def __iadd__(self, module):
        """Attach the child 'module' at the next available attachment point"""
        for conn in self.available:
            self[conn] = module
            return self
        raise NoAvailable("There were no available or non-obstructed free\
                spaces")

    def __repr__(self):
        """Create print friendly representation of this module"""
        return ("{!s}(children: {:d}, len: {:d}, available: {:d})"
                .format(self.__class__.__name__,
                        len(self.children),
                        len(self),
                        len(self.available)))

    def __str__(self):
        """Create a string friendly representation of this module"""
        res = ""
        self_name = self.__class__.__name__
        if self.children:
            length = int(len(self.children) / 2.)
            for i, child in enumerate(self.children):
                if i != length:
                    child_str = ' ' * len(self_name) + '  |-- ' + str(child)
                else:
                    child_str = self_name + ' -|-- ' + str(child)
                res += child_str
        else:
            res = self_name + '\n'
        return res

    def _update(self, parent=None, pos=None, direction=None):
        """Update configuration for the module.

        This function should update the parent pointer and internal position of
        the module. The 'pos' argument is the central point where this module
        connects to parent. The 'direction' argument is a vector pointing in
        the direction the module should be attached at.

        If no arguments are given it indicates an update back to center
        position."""
        if (parent is not None
                and self.parent is not None
                and parent != self.parent):
            raise ModuleAttached("This module already has a parent: {}"
                                 .format(self.parent))
        self.parent = parent
        # NOTE: We accept 'None' parents which could indicate a detachment of
        # this module
        if parent is not None:
            # The below equation rotates the 'connection_axis' parameter to
            # look in the same direction as 'direction'
            v = np.cross(self.connection_axis, direction)
            c = self.connection_axis.dot(direction)
            skew = np.array([[0., -v[2], v[1]],
                             [v[2], 0., -v[0]],
                             [-v[1], v[0], 0.]])
            if 1. + c != 0.:
                orient = Rot(np.identity(3) + skew
                             + skew.dot(skew) * (1. / (1. + c)))
            else:
                # This means that 'connection_axis' and 'direction' point in
                # the same direction, to convert we can simply flip one axis
                # and rotate by pi
                orient = Rot.from_axis(np.flip(self.connection_axis), np.pi)
            # Update own orientation
            # self.orientation += parent.orientation + orient
            self.orientation += orient

    def update_children(self):
        """Update all child modules of self"""
        raise NotImplementedError("Not supported")

    def spawn(self):
        """Spawn the module in the physics simulation"""
        raise NotImplementedError("Not supported")
