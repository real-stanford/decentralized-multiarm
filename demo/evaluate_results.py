import numpy as np
import pandas as pd
import json
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--result_dir', required=True, type=str)
    args = parser.parse_args()
    return args


def evaluate_results(df):
    stats = {}
    df_timeout = df.loc[df['limit'] == df['step_count']]
    df_success = df.loc[df['success']]
    indices = []
    for info in df['info']:
        if type(info) is str:
            info = eval(info)
            indices.append(info[1] == 'ur5_robotiq' and info[2] == 'ur5_robotiq')
        else:
            indices.append(False)
    df_robot_collision = df.loc[indices]

    indices = []
    for info in df['info']:
        if type(info) is str:
            info = eval(info)
            indices.append(info[2] == 'plane')
        else:
            indices.append(False)
    df_plane_collision = df.loc[indices]
    n_valid_exps = len(df_success) + len(df_robot_collision) + len(df_plane_collision)

    stats['num_exps'] = len(df)
    stats['num_valid_exps'] = n_valid_exps
    stats['num_success'] = len(df_success)
    stats['avg_steps'] = df_success.mean().step_count
    stats['num_plane_collision'] = len(df_plane_collision)
    stats['num_robot_collision'] = len(df_robot_collision)
    stats['num_timeout'] = len(df_timeout)
    stats['success_rate'] = stats['num_success'] / stats['num_valid_exps']
    return stats

if __name__ == "__main__":
    args = get_args()
    result_filepath = os.path.join(args.result_dir, 'results.csv')
    df = pd.read_csv(result_filepath, index_col=0)
    stats = evaluate_results(df)
    print(json.dumps(stats, indent=4))