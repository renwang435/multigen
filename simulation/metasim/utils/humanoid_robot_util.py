"""Helper functions for humanoid robots, including h1 and h1_simple_hand."""

import torch

from metasim.utils.math import axis_angle_from_quat, matrix_from_quat, quat_from_angle_axis, quat_mul


def torso_upright(envstate, robot_name: str):
    """Returns the projection of the torso's z-axis onto the world's z-axis.

    Args:
        envstate (dict): Environment state dictionary.
        robot_name (str): Name of the robot.

    Returns:
        float: The projection value, shape=(1,)
    """
    quat = envstate["robots"][robot_name]["body"]["pelvis"]["rot"]  # (4,)
    xmat = matrix_from_quat(torch.tensor(quat).unsqueeze(0))[0]  # (3, 3)
    return xmat[2, 2].item()


def torso_upright_tensor(envstate, robot_name: str):
    """Returns the projection of the torso's z-axis onto the world's z-axis for a batch of environments.

    Args:
        envstate: Environment state object with batched robot states.
        robot_name (str): Name of the robot.

    Returns:
        torch.Tensor: Projection values, shape=(batch_size,)
    """
    robot_body_name = envstate.robots[robot_name].body_names
    body_id = robot_body_name.index("pelvis")
    quat = envstate.robots[robot_name].body_state[:, body_id, 3:7]  # (batch_size, 4)
    xmat = matrix_from_quat(quat)  # (batch_size, 3, 3)
    return xmat[:, 2, 2]


def head_height(envstate, robot_name: str):
    """Returns the height of the head, actually the neck."""
    raise NotImplementedError("head_height is not implemented for isaacgym and isaaclab")
    # return envstate["robots"][robot_name]["head"]["pos"][2]  # Good for mujoco, but isaacgym and isaaclab don't have head site


def neck_height(envstate, robot_name: str):
    """Returns the height of the neck."""
    return (
        envstate["robots"][robot_name]["body"]["left_shoulder_roll_link"]["pos"][2]
        + envstate["robots"][robot_name]["body"]["right_shoulder_roll_link"]["pos"][2]
    ) / 2


def neck_height_tensor(envstate, robot_name: str):
    """Returns the height of the neck."""
    robot_body_name = envstate.robots[robot_name].body_names
    body_id_l = robot_body_name.index("left_shoulder_roll_link")
    body_id_r = robot_body_name.index("right_shoulder_roll_link")
    body_pos_l = envstate.robots[robot_name].body_state[:, body_id_l, 2]
    body_pos_r = envstate.robots[robot_name].body_state[:, body_id_r, 2]
    return (body_pos_l + body_pos_r) / 2


def left_foot_height(envstate, robot_name: str):
    """Returns the height of the left foot."""
    # return envstate[f"{_METASIM_SITE_PREFIX}left_foot"]["pos"][2] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["left_ankle_link"]["pos"][2]


def right_foot_height(envstate, robot_name: str):
    """Returns the height of the right foot."""
    # return envstate[f"{_METASIM_SITE_PREFIX}right_foot"]["pos"][2] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["right_ankle_link"]["pos"][2]


def robot_position(envstate, robot_name: str):
    """Returns position of the robot."""
    return envstate["robots"][robot_name]["pos"]


def robot_position_tensor(envstate, robot_name: str):
    """Returns position of the robot."""
    return envstate.robots[robot_name].root_state[:, 0:3]


def robot_velocity(envstate, robot_name: str):
    """Returns the velocity of the robot."""
    return envstate["robots"][robot_name]["vel"]


def robot_velocity_tensor(envstate, robot_name: str):
    """Returns the velocity of the robot."""
    return envstate.robots[robot_name].root_state[:, 7:10]


def robot_local_velocity_tensor(envstate, robot_name: str):
    """Returns the local linear velocity of the robot in its local frame.

    Args:
        envstate: Environment state object with batched robot states.
        robot_name (str): Name of the robot.

    Returns:
        torch.Tensor: Local velocities, shape=(batch_size, 2), where columns correspond to forward (x) and lateral (y) velocities.
    """
    world_velocity = robot_velocity_tensor(envstate, robot_name)
    world_rotation = robot_rotation_tensor(envstate, robot_name)

    # Extract yaw (rotation around z-axis) from the full rotation matrices
    def decompose_rotation_with_zaxis(rot_quat):
        """Decompose a rotation quaternion into a z angle and a xy rotation. R = Rz * Rxy.

        Args:
            rot_quat: The rotation quaternion. Shape is (batch_size, 4).

        Returns:
            z_angle: The z angle. Shape is (batch_size,).
        """
        rot_matrix = matrix_from_quat(rot_quat)  # Compute rotation matrix
        batch_size = rot_matrix.shape[0]
        Rz_angle = torch.zeros(batch_size, device=rot_matrix.device)
        for i in range(batch_size):
            z_new = rot_matrix[i] @ torch.tensor(
                [0.0, 0.0, 1.0], dtype=torch.float, device=rot_matrix.device
            )  # Compute z-axis rotation direction
            z_ori = torch.tensor(
                [0.0, 0.0, 1.0], dtype=torch.float, device=rot_matrix.device
            )  # z-axis original direction
            theta_z = torch.arccos(
                torch.dot(z_new, z_ori) / (torch.norm(z_new) * torch.norm(z_ori))
            )  # Compute z-axis rotation angle
            rot_axis_z = torch.cross(z_new, z_ori, dim=0) / torch.norm(
                torch.cross(z_new, z_ori, dim=0)
            )  # Compute z-axis rotation axis
            Rz_quat = quat_mul(
                quat_from_angle_axis(theta_z, rot_axis_z)[None, :], rot_quat[i][None, :]
            )  # Compute z-axis rotation quaternion
            Rz_rotvec = axis_angle_from_quat(Rz_quat)  # Compute z-axis rotation vector
            # print(f"Rz_rotvec: {Rz_rotvec/torch.norm(Rz_rotvec, dim=-1)}")
            # from metasim.utils.math import quat_inv
            # Rz_quat_inv = quat_inv(Rz_quat)
            # Rxy_quat = quat_mul(Rz_quat_inv, rot_quat)
            # Rxy_rotvec = axis_angle_from_quat(Rxy_quat)
            # print(f"Rxy_rotvec: {Rxy_rotvec/torch.norm(Rxy_rotvec, dim=-1)}")
            # exit()
            Rz_angle[i] = torch.norm(Rz_rotvec, dim=-1)  # Compute z-axis rotation angle
        return Rz_angle

    yaws = decompose_rotation_with_zaxis(world_rotation)
    cos_z = torch.cos(yaws)
    sin_z = torch.sin(yaws)
    local_xy_velocity = torch.zeros((world_velocity.shape[0], 2), device=world_velocity.device)
    local_xy_velocity[:, 0] = world_velocity[:, 0] * cos_z + world_velocity[:, 1] * sin_z
    local_xy_velocity[:, 1] = -world_velocity[:, 0] * sin_z + world_velocity[:, 1] * cos_z
    return local_xy_velocity


def robot_rotation(envstate, robot_name: str):
    """Returns the rotation of the robot."""
    return envstate["robots"][robot_name]["rot"]


def robot_rotation_tensor(envstate, robot_name: str):
    """Returns the rotation of the robot."""
    return envstate.robots[robot_name].root_state[:, 3:7]


def torso_vertical_orientation(envstate, robot_name: str):
    """Returns the z-projection of the torso orientation matrix.

    Args:
        envstate: Environment state object with batched robot states.
        robot_name (str): Name of the robot.

    Returns:
        torch.Tensor: z-axis projection, shape=(batch_size,)
    """
    quat = envstate["robots"][robot_name]["body"]["pelvis"]["rot"]  # (4,)
    xmat = matrix_from_quat(torch.tensor(quat).unsqueeze(0))[0]  # (3, 3)
    return xmat[2, :]


def actuator_forces(envstate, robot_name: str):
    """Returns a copy of the forces applied by the actuators."""
    return (
        torch.tensor([x for x in envstate["robots"][robot_name]["dof_torque"].values()])
        if envstate["robots"][robot_name].get("dof_torque", None) is not None
        else torch.zeros(len(envstate["robots"][robot_name]["dof_pos"]))
    )


def actuator_forces_tensor(envstate, robot_name: str):
    """Returns a copy of the forces applied by the actuators."""
    return (
        envstate.robots[robot_name].joint_effort_target
        if envstate.robots[robot_name].joint_effort_target is not None
        else torch.zeros_like(envstate.robots[robot_name].joint_pos)
    )


def left_hand_position(envstate, robot_name: str):
    """Returns the position of the left hand."""
    # return envstate[f"{_METASIM_SITE_PREFIX}left_hand"]["pos"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["left_elbow_link"]["pos"]


def left_hand_velocity(envstate, robot_name: str):
    """Returns the velocity of the left hand."""
    # return envstate[f"{_METASIM_BODY_PREFIX}left_hand"]["left_hand_subtreelinvel"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["left_elbow_link"]["vel"]


def left_hand_orientation(envstate, robot_name: str):
    """Returns the orientation of the left hand."""
    # return envstate[f"{_METASIM_SITE_PREFIX}left_hand"]["rot"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["left_elbow_link"]["rot"]


def right_hand_position(envstate, robot_name: str):
    """Returns the position of the right hand."""
    # return envstate[f"{_METASIM_SITE_PREFIX}right_hand"]["pos"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["right_elbow_link"]["pos"]


def right_hand_velocity(envstate, robot_name: str):
    """Returns the velocity of the right hand."""
    # return envstate[f"{_METASIM_BODY_PREFIX}right_hand"]["right_hand_subtreelinvel"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["right_elbow_link"]["vel"]


def right_hand_orientation(envstate, robot_name: str):
    """Returns the orientation of the right hand."""
    # return envstate[f"{_METASIM_SITE_PREFIX}right_hand"]["rot"] # Only for mujoco
    return envstate["robots"][robot_name]["body"]["right_elbow_link"]["rot"]
