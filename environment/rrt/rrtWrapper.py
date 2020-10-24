from .pybullet_utils import (
    configure_pybullet, draw_line, remove_all_markers
)
from itertools import chain
import ray
import pybullet as p
from .ur5_group import UR5Group
from time import sleep
from .rrt_connect import birrt
import pickle


@ray.remote
class RRTWrapper:
    def __init__(self, env_config, gui=False):
        from environment.utils import create_ur5s, Target
        print("[RRTWrapper] Setting up RRT supervision")
        # set up simulator
        configure_pybullet(
            rendering=gui,
            debug=False,
            yaw=0, pitch=0,
            dist=1.0,
            target=(0, 0, 0.3))
        self.gui = gui
        plane = p.loadURDF(
            "plane.urdf",
            [0, 0, -env_config['collision_distance'] - 0.01])
        self.obstacles = [plane]

        def create_ur5s_fn():
            return create_ur5s(
                radius=0.8,
                count=env_config['max_ur5s_count'],
                speed=env_config['ur5_speed'])

        self.ur5_group = UR5Group(
            create_ur5s_fn=create_ur5s_fn,
            collision_distance=env_config['collision_distance'])
        self.targets = [Target(
            pose=[[0, 0, 0], [0, 0, 0, 1]],
            color=ur5.color)
            for ur5 in self.ur5_group.all_controllers]
        self.actor_handle = None

    def set_actor_handle(self, actor_handle):
        self.actor_handle = actor_handle

    def birrt_from_task(self, task):
        return self.birrt(
            start_conf=task.start_config,
            goal_conf=task.start_goal_config,
            ur5_poses=task.base_poses,
            target_eff_poses=task.target_eff_poses)

    def birrt_from_task_with_actor_handle(self, task):
        rv = self.birrt_from_task(task)
        return rv, task.id, self.actor_handle

    def birrt(self, start_conf, goal_conf,
              ur5_poses, target_eff_poses,
              resolutions=0.1, timeout=100000):
        if self.gui:
            remove_all_markers()
            for pose, target in zip(target_eff_poses, self.targets):
                target.set_pose(pose)
        self.ur5_group.setup(ur5_poses, start_conf)
        extend_fn = self.ur5_group.get_extend_fn(resolutions)
        collision_fn = self.ur5_group.get_collision_fn()
        start_conf = list(chain.from_iterable(start_conf))
        goal_conf = list(chain.from_iterable(goal_conf))
        path = birrt(start_conf=start_conf,
                     goal_conf=goal_conf,
                     distance=self.ur5_group.distance_fn,
                     sample=self.ur5_group.sample_fn,
                     extend=extend_fn,
                     collision=collision_fn,
                     iterations=10000,
                     smooth=5,
                     visualize=self.gui,
                     fk=self.ur5_group.forward_kinematics,
                     group=True,
                     greedy=True,
                     timeout=timeout)
        if path is None:
            return None
        if self.gui:
            self.demo_path(path)
        return path

    def demo_path(self, path_conf):
        edges = []
        for i in range(len(path_conf)):
            if i != len(path_conf) - 1:
                for pose1, pose2 in zip(
                        self.ur5_group.forward_kinematics(
                            path_conf[i]),
                        self.ur5_group.forward_kinematics(
                            path_conf[i + 1])):
                    draw_line(pose1[0], pose2[0],
                              rgb_color=[1, 0, 0], width=6)
                    edges.append((pose1[0], pose2[0]))
        if self.record:
            with open('waypoints.pkl', 'wb') as f:
                pickle.dump(edges, f)
        for i, q in enumerate(path_conf):
            self.ur5_group.set_joint_positions(q)
            sleep(0.01)
