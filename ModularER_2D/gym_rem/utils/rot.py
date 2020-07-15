#!/usr/bin/env python

"""
Helper functions around rotations
"""
import numpy as np


class Rot(object):
    """
    This object represents a rotation. Use the static methods to construct and
    use the 'rotate' function to rotate vectors in 3 dimensions.
    """
    def __init__(self, mat):
        self._mat = np.asarray(mat)
        assert self._mat.shape == (3, 3), "Input matrix must be of shape\
                (3, 3) not: {}".format(self._mat.shape)

    def rotate(self, vector):
        """Rotate the vector by the rotation represented by this object"""
        return self._mat.dot(vector)

    @property
    def T(self):
        return Rot(self._mat.T)

    def __add__(self, rot):
        """Combine two rotations creating a new combined rotation of the two"""
        assert isinstance(rot, Rot)
        # return Rot(rot._mat @ self._mat)
        return Rot(self._mat @ rot._mat)

    def __iadd__(self, rot):
        """Add the rotation 'rot' to this rotation"""
        assert isinstance(rot, Rot)
        # self._mat = rot._mat @ self._mat
        self._mat = self._mat @ rot._mat
        return self

    def __repr__(self):
        roll, pitch, yaw = self.as_euler()
        return "Rot({:.2f}, {:.2f}, {:.2f})".format(roll, pitch, yaw)

    def as_euler(self):
        """Return the rotation represented as Euler angles"""
        roll = np.arctan2(self._mat[2, 0], self._mat[2, 1])
        pitch = np.arccos(self._mat[2, 2])
        yaw = -np.arctan2(self._mat[0, 2], self._mat[1, 2])
        return roll, pitch, yaw

    def as_axis(self):
        """Return the rotation represented as axis-angle"""
        tr = self._mat[0, 0] + self._mat[1, 1] + self._mat[2, 2]
        theta = np.arccos((tr - 1.) / 2.)
        div = 2. * np.sin(theta) if theta != 0 else 1.
        e1 = (self._mat[2, 1] - self._mat[1, 2]) / div
        e2 = (self._mat[0, 2] - self._mat[2, 0]) / div
        e3 = (self._mat[1, 0] - self._mat[0, 1]) / div
        return np.array([e1, e2, e3]), theta

    def as_quat(self):
        """Return the rotation represented as a quaternion [x, y, z, w]"""
        # The below conversions are gathered from:
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        tr = self._mat[0, 0] + self._mat[1, 1] + self._mat[2, 2]
        if tr > 0.0:
            s = 2. * np.sqrt(1. + tr)
            w = 0.25 * s
            x = (self._mat[2, 1] - self._mat[1, 2]) / s
            y = (self._mat[0, 2] - self._mat[2, 0]) / s
            z = (self._mat[1, 0] - self._mat[0, 1]) / s
        elif (self._mat[0, 0] > self._mat[1, 1]
              and self._mat[0, 0] > self._mat[2, 2]):
            s = 2. * np.sqrt(1. + self._mat[0, 0] - self._mat[1, 1]
                             - self._mat[2, 2])
            w = (self._mat[2, 1] - self._mat[1, 2]) / s
            x = 0.25 * s
            y = (self._mat[0, 1] + self._mat[1, 0]) / s
            z = (self._mat[0, 2] + self._mat[2, 0]) / s
        elif self._mat[1, 1] > self._mat[2, 2]:
            s = 2. * np.sqrt(1. + self._mat[1, 1] - self._mat[0, 0]
                             - self._mat[2, 2])
            w = (self._mat[0, 2] - self._mat[2, 0]) / s
            x = (self._mat[0, 1] + self._mat[1, 0]) / s
            y = 0.25 * s
            z = (self._mat[1, 2] + self._mat[2, 1]) / s
        else:
            s = 2. * np.sqrt(1. + self._mat[2, 2] - self._mat[0, 0]
                             - self._mat[1, 1])
            w = (self._mat[1, 0] - self._mat[0, 1]) / s
            x = (self._mat[0, 2] + self._mat[2, 0]) / s
            y = (self._mat[1, 2] + self._mat[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w])

    @staticmethod
    def from_axis(axis, angle):
        """Create a rotation from the given axis-angle representation"""
        mat = np.identity(3) * np.cos(angle)
        mat += np.sin(angle) * np.array([[0., -axis[2], axis[1]],
                                         [axis[2], 0., -axis[0]],
                                         [-axis[1], axis[0], 0.]])
        mat += (1. - np.cos(angle)) * (np.outer(axis, axis))
        return Rot(mat)

    @staticmethod
    def from_euler(roll, pitch, yaw):
        """Create a rotation from the given RPY representation"""
        a_x = np.array([[1., 0., 0.],
                        [0., np.cos(roll), -np.sin(roll)],
                        [0., np.sin(roll), np.cos(roll)]])
        a_y = np.array([[np.cos(pitch), 0., np.sin(pitch)],
                        [0., 1., 0.],
                        [-np.sin(pitch), 0., np.cos(pitch)]])
        a_z = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                        [np.sin(yaw), np.cos(yaw), 0.],
                        [0., 0., 1.]])
        return Rot((a_z @ a_y) @ a_x)

    @staticmethod
    def from_quat(quat):
        """Create a rotation from the given quaternion representation.

        This assumes that 'quat' is a vector [x, y, z, w]"""
        q_dot = quat[:-1]
        q = np.array([[0., -quat[2], quat[1]],
                      [quat[2], 0., -quat[0]],
                      [-quat[1], quat[0], 0.]])
        q_w = quat[-1]
        mat = (q_w ** 2 - q_dot.T.dot(q_dot)) * np.identity(3)
        mat += 2. * q_dot.dot(q_dot.T)
        mat += 2. * q_w * q
        return Rot(mat)

    @staticmethod
    def identity():
        """Create a default rotation"""
        return Rot(np.identity(3))
