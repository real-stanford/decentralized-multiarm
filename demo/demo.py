import pybullet as p
from misc_utils import configure_pybullet, TargetObject, ConfigInfo, write_csv_line
from ur5_robotiq_controller import UR5RobotiqPybulletController
import os
import pybullet_utils as pu
import pickle
import numpy as np
from executer import Executer
import threading
from recorder import PybulletRecorder
from time import sleep
from random import seed as random_seed, uniform, choice
from task import (
    UR5AsyncTaskRunner as TaskRunner,
    PolicyTask, ControlArmTask,
    CloseGripperTask, AttachToGripperTask,
    CartesianControlTask, OpenGripperTask,
    DetachToGripperTask, SetTargetTask
)
from policy import InferencePolicy
from collections import namedtuple
from itertools import combinations
from misc_utils import OtherObject
from math import radians, degrees
import ray
import time
from tqdm import tqdm

ConfigInfo = namedtuple(
    'ConfigInfo', ['joint_config', 'ee_pose', 'pos_distance', 'quat_distance'])

d_robot = 0.6
d_target = 1.0
robot_base_poses = [[[d_robot, d_robot, 0.01], [0, 0, 0, 1]],
                    [[d_robot, -d_robot, 0.01], [0, 0, 0, 1]],
                    [[-d_robot, d_robot, 0.01],
                     pu.quaternion_from_euler([0, 0, np.pi])],
                    [[-d_robot, -d_robot, 0.01],
                     pu.quaternion_from_euler([0, 0, np.pi])]]
dump_positions = [[0.2, 0.2, 0.5],
                  [0.2, -0.2, 0.5],
                  [-0.2, 0.2, 0.5],
                  [-0.2, -0.2, 0.5]]
target_xys = [[d_target, 0],
              [0, -d_target],
              [0, d_target],
              [-d_target, 0]]


def set_initial_random_configs(ur5s, randomization_magnitude=0.4):
    # to be safe from plane, constrain the arm eef higher
    above_threshold = 0.2
    while True:
        for ur5 in ur5s:
            ur5.reset()
            curr = np.array(ur5.get_arm_joint_values())
            ur5.set_arm_joints(
                curr +
                np.array([uniform(
                    -randomization_magnitude,
                    randomization_magnitude)
                    for _ in range(6)])
            )

        if not any([ur5.check_collision()[0] for ur5 in ur5s]) and \
                all([ur5.get_eef_pose()[0][2] > above_threshold for ur5 in ur5s]):
            break


def prepare_task_runners(ur5s, targets, policy):
    task_runners = []
    # for each robot there is multiple targets
    for i, (ur5, robot_targets) in enumerate(zip(ur5s, targets)):
        folder_path = os.path.join('tasks', 'robot' + str(i))
        grasps = [pickle.load(open(os.path.join(
            folder_path, t.object_name, 'grasp.p'), 'rb')) for t in robot_targets]
        grasp_configs = [pickle.load(open(os.path.join(folder_path, t.object_name, 'grasp_config.p'), 'rb')) for t in
                         robot_targets]
        dump_configs = [pickle.load(open(os.path.join(folder_path, t.object_name, 'dump_config.p'), 'rb')) for t in
                        robot_targets]
        dump_jvs = [pickle.load(open(os.path.join(folder_path, t.object_name, 'dump_jv.p'), 'rb')) for t in
                    robot_targets]
        initial_poses = [pickle.load(open(os.path.join(folder_path, t.object_name, 'initial_pose.p'), 'rb')) for t in
                         robot_targets]

        ur5_tasks = []
        for target, grasp, grasp_config, dump_config, dump_jv, initial_pose in zip(robot_targets, grasps, grasp_configs,
                                                                                   dump_configs,
                                                                                   dump_jvs, initial_poses):
            # pu.create_frame_marker(grasp.pre_grasp_pose)
            ur5_single_target_task = [
                SetTargetTask(ur5=ur5,
                              target_id=target.id,
                              initial_pose=initial_pose),
                PolicyTask(ur5=ur5,
                           target_pose=grasp_config.ee_pose,
                           policy=policy,
                           position_tolerance=0.05,
                           orientation_tolerance=0.2),
                ControlArmTask(
                    ur5=ur5,
                    target_config=grasp.pre_grasp_jv
                ),
                ControlArmTask(
                    ur5=ur5,
                    target_config=grasp.grasp_jv
                ),
                CloseGripperTask(ur5=ur5),
                AttachToGripperTask(ur5=ur5, target_id=target.id),
                CartesianControlTask(ur5=ur5, axis='z', value=0.3),
                PolicyTask(ur5=ur5,
                           target_pose=dump_config.ee_pose,
                           policy=policy,
                           position_tolerance=0.05,
                           orientation_tolerance=0.2),
                ControlArmTask(
                    ur5=ur5,
                    target_config=dump_jv
                ),
                CartesianControlTask(ur5=ur5, axis='z', value=-0.05),
                DetachToGripperTask(ur5=ur5),
                OpenGripperTask(ur5=ur5),
                CartesianControlTask(ur5=ur5, axis='z', value=0.05),
            ]
            ur5_tasks += ur5_single_target_task
        task_runners.append(TaskRunner(ur5=ur5, tasks=ur5_tasks + [ControlArmTask(
            ur5=ur5,
            target_config=ur5.RESET
        )]))
    return task_runners


def create_target_xyss(d_target, delta, num_objects):
    result = []
    result.append([[d_target + delta * i, 0] for i in range(num_objects)])
    result.append([[0, -d_target - delta * i] for i in range(num_objects)])
    result.append([[0, d_target + delta * i] for i in range(num_objects)])
    result.append([[-d_target - delta * i, 0] for i in range(num_objects)])
    return result


def create_scene(random=True,
                 target_object_names=None,
                 initial_configs=None,
                 num_targets_per_arm=1):
    plane = p.loadURDF('plane.urdf')
    plastic_bin = OtherObject("assets/tote/tote.urdf",
                              initial_pose=[[0, 0, 0], [0, 0, 0, 1]])

    if target_object_names is None:
        pickable_objects = [os.listdir(os.path.join('tasks', robot))
                            for robot in
                            ['robot0', 'robot1', 'robot2', 'robot3']]
        target_object_names = [choice(
            list((combinations(po, num_targets_per_arm))))
            for po in pickable_objects]

    target_xyss = create_target_xyss(
        d_target, delta=0.2, num_objects=num_targets_per_arm)

    ur5s = [UR5RobotiqPybulletController(
        base_pose=rb) for rb in robot_base_poses]
    [p.addUserDebugText(str(i),
                        [pose[0][0], pose[0][1], 0.6],
                        (1, 0, 0),
                        textSize=2)
     for i, pose in enumerate(robot_base_poses)]
    targets = []
    for i, (names, target_xys) in enumerate(
            zip(target_object_names, target_xyss)):
        initial_poses = [pickle.load(open(
            os.path.join('tasks', 'robot' + str(i),
                         n, 'initial_pose.p'), 'rb'))
                         for n in names]
        rpys = [[degrees(e) for e in pu.euler_from_quaternion(pose[1])]
                for pose in initial_poses]
        ur5_targets = [TargetObject(n, target_xy, 0, rpy)
                       for n, target_xy, rpy in zip(names, target_xys, rpys)]
        targets.append(ur5_targets)

    if random and initial_configs is None:
        set_initial_random_configs(ur5s)
    else:
        for ur5, c in zip(ur5s, initial_configs):
            ur5.set_arm_joints(c)

    # step for a few seconds
    pu.step(3)
    return ur5s, targets, plastic_bin, dump_positions


def check_success(targets, bin):
    bbox = p.getAABB(bin.id, -1)
    target_in_bins = []
    for tt in targets:
        for t in tt:
            target_pose = t.get_pose()
            target_in_bin = bbox[0][0] < target_pose[0][0] < bbox[1][0] and \
                bbox[0][1] < target_pose[0][1] < bbox[1][1]
            target_in_bins.append(target_in_bin)
    return all(target_in_bins)


@ray.remote(num_cpus=1)
def demo_with_seed(seed, video_dir, result_dir, recorder_dir, load, record_video=False):
    configure_pybullet(rendering=False, debug=False)
    random_seed(seed)
    result_filepath = os.path.join(result_dir, 'results.csv')
    if record_video:
        video_filepath = os.path.join(video_dir, '{}.mp4'.format(seed))
        logging = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4, video_filepath)

    if not load:
        ur5s, targets, plastic_bin, plane = create_scene()
        target_object_names = [[t.object_name for t in ts]
                               for ts in targets]
        intitial_configs = [ur5.get_arm_joint_values() for ur5 in ur5s]
        benchmark_dir = os.path.join('tasks', 'benchmark', str(seed))
        if not os.path.exists(benchmark_dir):
            os.makedirs(benchmark_dir)
        pickle.dump(target_object_names, open(os.path.join(
            benchmark_dir, 'target_object_names.p'), 'wb'))
        pickle.dump(intitial_configs, open(os.path.join(
            benchmark_dir, 'intitial_configs.p'), 'wb'))
    else:
        benchmark_dir = os.path.join('tasks', 'benchmark', str(seed))
        if not os.path.exists(benchmark_dir):
            raise ValueError('benchmark dataset does not exist!')
        target_object_names = pickle.load(
            open(os.path.join(benchmark_dir, 'target_object_names.p'), 'rb'))
        intitial_configs = pickle.load(
            open(os.path.join(benchmark_dir, 'intitial_configs.p'), 'rb'))
        ur5s, targets, plastic_bin, plane = create_scene(
            target_object_names=target_object_names,
            initial_configs=intitial_configs)

    # for rendering
    recorder = PybulletRecorder()
    for ur5 in ur5s:
        recorder.register_object(
            body_id=ur5.id,
            path='assets/ur5/ur5_robotiq.urdf'
        )
    for ur5_targets in targets:
        for target in ur5_targets:
            recorder.register_object(
                body_id=target.id,
                path=target.urdf_path
            )
    recorder.register_object(
        body_id=plastic_bin.id,
        path=plastic_bin.urdf
    )

    policy = InferencePolicy()
    task_runners = prepare_task_runners(
        ur5s=ur5s,
        targets=targets,
        policy=policy)
    executer = Executer(task_runners=task_runners,
                        recorder=recorder, recorder_dir=recorder_dir)
    executer_success, step_count, info = executer.run()
    success = False if not executer_success else check_success(
        targets, plastic_bin)
    if record_video:
        p.stopStateLogging(logging)
    p.disconnect()

    result = {
        'exp': seed,
        'success': success,
        'limit': executer.limit,
        'step_count': step_count,
        'info': info,
        'simulation_output_path':
        executer.simulation_output_path
    }
    write_csv_line(result_filepath, result)
    return result, executer.simulation_output_path


if __name__ == "__main__":
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    video_dir = 'videos' + '_' + timestr
    result_dir = 'results' + '_' + timestr
    recorder_dir = 'simulation' + '_' + timestr
    record_video = False
    load = True
    if not os.path.exists(video_dir) and record_video:
        os.makedirs(video_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(recorder_dir):
        os.makedirs(recorder_dir)
    num_exps = 500
    ray.init(logging_level=ray.logging.FATAL,
             log_to_driver=False)
    tasks = [demo_with_seed.remote(
        seed=i, video_dir=video_dir,
        result_dir=result_dir, recorder_dir=recorder_dir,
        load=load, record_video=record_video)
        for i in range(num_exps)]
    num_valids = 0
    num_successes = 0
    with tqdm(total=num_exps, dynamic_ncols=True, desc='Running Demo') as pbar:
        while len(tasks) > 0:
            done, tasks = ray.wait(tasks, num_returns=1)
            result, simulation_pkl_path = ray.get(done[0])
            info = result['info']
            if info is not None:
                # collide with other robots
                robot_collided = info[1] == 'ur5_robotiq' and info[2] == 'ur5_robotiq'
                # collide with plane
                robot_collided = robot_collided or info[2] == 'plane'
                is_valid = robot_collided or result['success']
            else:
                is_valid = True
            num_successes += int(result['success'])
            num_valids += int(is_valid)
            if is_valid:
                print(f'Exp {result["exp"]}:')
                print(f'\tSuccess: {result["success"]}')
                print(f'\tPath: {simulation_pkl_path}')
            pbar.update()
            if num_valids > 0:
                pbar.set_description(
                    f'Success Rate: {num_successes/num_valids:.04f}')
