# ------------------------------------------------
# This file is partly borrowed and adapted from 
# https://github.com/caelan/motion-planners
# ------------------------------------------------


from .smoothing import smooth_path
from .rrt import TreeNode, configs
from .rrt_utils import irange, argmin
from .pybullet_utils import draw_line
from time import time
import pickle


def rrt_connect(q1,
                q2,
                distance,
                sample,
                extend,
                collision,
                iterations,
                visualize,
                fk,
                group,
                greedy,
                timeout,
                record=False):
    start_time = time()
    if collision(q1) or collision(q2):
        return None
    root1, root2 = TreeNode(q1), TreeNode(q2)
    nodes1, nodes2 = [root1], [root2]
    edges = []
    if visualize:
        color1, color2 = [0, 1, 0], [0, 0, 1]
    for _ in irange(iterations):
        if float(time() - start_time) > timeout:
            break
        if len(nodes1) > len(nodes2):
            nodes1, nodes2 = nodes2, nodes1
            if visualize:
                color1, color2 = color2, color1
        s = sample()

        last1 = argmin(lambda n: distance(n.config, s), nodes1)
        for q in extend(last1.config, s):
            if collision(q):
                break
            if visualize:
                assert fk is not None, 'please provide a fk when visualizing'
                if group:
                    for pose_now, pose_prev in zip(fk(q), fk(last1.config)):
                        draw_line(
                            pose_prev[0], pose_now[0], rgb_color=color1, width=1)
                        if record:
                            edges.append((pose_prev[0], pose_now[0]))
                else:
                    p_now = fk(q)[0]
                    p_prev = fk(last1.config)[0]
                    draw_line(p_prev, p_now, rgb_color=color1, width=1)
                    if record:
                        edges.append((p_prev, p_now))
            last1 = TreeNode(q, parent=last1)
            nodes1.append(last1)
            if not greedy:
                break

        last2 = argmin(lambda n: distance(n.config, last1.config), nodes2)
        for q in extend(last2.config, last1.config):
            if collision(q):
                break
            if visualize:
                assert fk is not None, 'please provide a fk when visualizing'
                if group:
                    for pose_now, pose_prev in zip(fk(q), fk(last2.config)):
                        draw_line(
                            pose_prev[0], pose_now[0], rgb_color=color2, width=1)
                        if record:
                            edges.append((pose_prev[0], pose_now[0]))
                else:
                    p_now = fk(q)[0]
                    p_prev = fk(last2.config)[0]
                    draw_line(p_prev, p_now, rgb_color=color2, width=1)
                    if record:
                        edges.append((p_prev, p_now))
            last2 = TreeNode(q, parent=last2)
            nodes2.append(last2)
            if not greedy:
                break
        if last2.config == last1.config:
            path1, path2 = last1.retrace(), last2.retrace()
            if path1[0] != root1:
                path1, path2 = path2, path1
            if record:
                with open('edges.pkl', 'wb') as f:
                    pickle.dump(edges, f)
            return configs(path1[:-1] + path2[::-1])
    return None


def direct_path(q1, q2, extend, collision):
    if collision(q1) or collision(q2):
        return None
    path = [q1]
    for q in extend(q1, q2):
        if collision(q):
            return None
        path.append(q)
    return path


def birrt(start_conf,
          goal_conf,
          distance,
          sample,
          extend,
          collision,
          iterations,
          smooth,
          visualize,
          fk,
          group,
          greedy,
          timeout):
    path = direct_path(start_conf, goal_conf, extend, collision)
    if path is not None:
        return path
    path = rrt_connect(start_conf,
                       goal_conf,
                       distance,
                       sample,
                       extend,
                       collision,
                       iterations,
                       visualize,
                       fk,
                       group,
                       greedy,
                       timeout)
    if path is not None:
        return smooth_path(path, extend, collision, iterations=smooth)
    return None