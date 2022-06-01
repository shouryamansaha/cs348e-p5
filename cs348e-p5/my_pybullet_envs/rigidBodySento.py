#  Copyright 2020 Stanford University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pybullet as p
import numpy as np

GRAVITY = 9.81
WATER_SURFACE = 2.0
WATER_DENSITY = 1010
# WATER_DENSITY = 500
DRAG_COEFF = 300


def create_primitive_shape(mass, shape, dim, color=(0.6, 0, 0, 1), collidable=True, init_xyz=(0, 0, 0),
                           init_quat=(0, 0, 0, 1)):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder, (radius) for sphere
    # init_xyz vec3 being initial obj location, init_quat being initial obj orientation
    visual_shape_id = None
    collision_shape_id = -1
    if shape == p.GEOM_BOX:
        visual_shape_id = p.createVisualShape(shapeType=shape, halfExtents=dim, rgbaColor=color)
        if collidable:
            collision_shape_id = p.createCollisionShape(shapeType=shape, halfExtents=dim)
    elif shape == p.GEOM_CYLINDER:
        visual_shape_id = p.createVisualShape(shape, dim[0], [1, 1, 1], dim[1], rgbaColor=color)
        if collidable:
            collision_shape_id = p.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
    elif shape == p.GEOM_SPHERE:
        visual_shape_id = p.createVisualShape(shape, radius=dim[0], rgbaColor=color)
        if collidable:
            collision_shape_id = p.createCollisionShape(shape, radius=dim[0])

    sid = p.createMultiBody(baseMass=mass, baseInertialFramePosition=[0, 0, 0],
                            baseCollisionShapeIndex=collision_shape_id,
                            baseVisualShapeIndex=visual_shape_id,
                            basePosition=init_xyz, baseOrientation=init_quat)
    return sid


def get_link_com_xyz_orn(body_id, link_id):
    # get the world transform (xyz and quaternion) of the Center of Mass of the link
    # We *assume* link CoM transform == link shape transform (the one you use to calculate fluid force on each shape)
    assert link_id >= -1
    if link_id == -1:
        link_com, link_quat = p.getBasePositionAndOrientation(body_id)
    else:
        link_com, link_quat, *_ = p.getLinkState(body_id, link_id, computeForwardKinematics=1)
    return list(link_com), list(link_quat)


def apply_external_world_force_on_local_point(body_id, link_id, world_force, local_com_offset):
    link_com, link_quat = get_link_com_xyz_orn(body_id, link_id)
    _, inv_link_quat = p.invertTransform([0., 0, 0], link_quat)  # obj->world
    local_force, _ = p.multiplyTransforms([0., 0, 0], inv_link_quat, world_force, [0, 0, 0, 1])
    p.applyExternalForce(body_id, link_id, local_force, local_com_offset, flags=p.LINK_FRAME)


def get_link_com_linear_velocity(body_id, link_id):
    # get the Link CoM linear velocity in the world coordinate frame
    assert link_id >= -1
    if link_id == -1:
        vel, _ = p.getBaseVelocity(body_id)
    else:
        vel = p.getLinkState(body_id, link_id, computeLinkVelocity=1, computeForwardKinematics=1)[6]
    return list(vel)


def get_dim_of_box_shape(body_id, link_id):
    # get the dimension (length-3 list of width, depth, height) of the input link
    # We *assume* each link is a box
    # p.getCollisionShapeData() might be useful to you
    # hint: Check out getCollisionShapeData function in PyBullet Tutorial Document
    dim = p.getCollisionShapeData(body_id, link_id)[0][3]
    return dim


def get_highest_and_lowest_points_world_xyz(body_id, link_id):
    # calculate and return the xyz of the highest
    # & lowest point of the link (box shaped) in world frame

    dim = get_dim_of_box_shape(body_id, link_id)
    local_corners = [[dim[0] / 2, dim[1] / 2, dim[2] / 2] for _ in range(8)]
    local_corners[1] = [-dim[0] / 2, dim[1] / 2, dim[2] / 2]
    local_corners[2] = [dim[0] / 2, -dim[1] / 2, dim[2] / 2]
    local_corners[3] = [dim[0] / 2, dim[1] / 2, -dim[2] / 2]
    local_corners[4] = [-dim[0] / 2, -dim[1] / 2, dim[2] / 2]
    local_corners[5] = [-dim[0] / 2, dim[1] / 2, -dim[2] / 2]
    local_corners[6] = [dim[0] / 2, -dim[1] / 2, -dim[2] / 2]
    local_corners[7] = [-dim[0] / 2, -dim[1] / 2, -dim[2] / 2]

    corners = [[0., 0, 0]] * 8
    pos, quat = get_link_com_xyz_orn(body_id, link_id)
    for j in range(8):
        corners[j], _ = p.multiplyTransforms(pos, quat, local_corners[j], [0, 0, 0, 1])
    zs = np.array(corners)[:, 2]
    high_xyz = corners[np.argmax(zs)]
    low_xyz = corners[np.argmin(zs)]

    return high_xyz, low_xyz


def get_face_normals(body_id, link_id):
    # calculate and return the unit normal vec of the 6 faces of the link, in world frame
    # NOTE! Transforming a vector from link to world frame is different from transforming a point on link!
    normals = [[0., 0, 0]] * 6

    local_normals = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    _, quat = get_link_com_xyz_orn(body_id, link_id)
    for j in range(6):
        normals[j], _ = p.multiplyTransforms([0., 0, 0], quat, local_normals[j], [0, 0, 0, 1])

    return normals


def normalize_velocity(lin_vel):
    # return normalized linear velocity (length-3 numpy array)
    norm = np.linalg.norm(lin_vel)
    if norm < 1e-6:
        return np.array(lin_vel)
    return np.array(lin_vel) / norm


def apply_fluid_force(body_id, given_forces=None):
    # apply fluid force to every link of the body using the formulae in the Jupyter tutorial

    if given_forces:
        for link_id, f_force in given_forces.items():
            apply_external_world_force_on_local_point(body_id, link_id, f_force, [0, 0, 0])
        return None

    forces = {}

    for link_id in range(-1, p.getNumJoints(body_id)):
        try:
            dim = get_dim_of_box_shape(body_id, link_id)
        except:
            continue
        high_xyz, low_xyz = get_highest_and_lowest_points_world_xyz(body_id, link_id)
        ratio = (WATER_SURFACE - low_xyz[2]) / (high_xyz[2] - low_xyz[2])
        ratio = np.clip(ratio, 0.0, 1.0)

        lift_force = [0, 0, GRAVITY * WATER_DENSITY * dim[0] * dim[1] * dim[2] * ratio]

        vel = get_link_com_linear_velocity(body_id, link_id)
        vel_dir = normalize_velocity(vel)
        normals = get_face_normals(body_id, link_id)

        areas = [dim[1] * dim[2], dim[1] * dim[2], dim[0] * dim[2], dim[0] * dim[2], dim[0] * dim[1],
                 dim[0] * dim[1]]
        scalars = [np.dot(np.array(normals[i]), vel_dir) * areas[i] for i in range(len(areas))]
        weight = -sum(x for x in scalars if x > 0) * DRAG_COEFF * ratio
        drag_force = [vel[i] * weight for i in range(3)]

        fluid_force = [lift_force[x] + drag_force[x] for x in range(3)]

        apply_external_world_force_on_local_point(body_id, link_id, fluid_force, [0, 0, 0])

        forces[link_id] = fluid_force

    return forces
