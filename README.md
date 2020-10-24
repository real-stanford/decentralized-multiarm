# Learning a Decentralized Multiarm Motion Planner

[Huy Ha](https://www.haquochuy.com),
[Jingxi Xu](https://jxu.ai/),
[Shuran Song](https://www.cs.columbia.edu/~shurans/),
<br>
Columbia University, New York, NY, United States<br>
[CoRL 2020](https://www.robot-learning.org/)

### [Project Page](https://multiarm.cs.columbia.edu/) | [Video](#) | [Paper](#) 

![](assets/teaser.png)

## Setup

Python 3.7 dependencies:
 - PyTorch 1.6.0
 - pybullet
 - numpy
 - numpy-quaternion
 - ray
 - tensorboardX

We've prepared a conda YAML file which contains all the necessary dependencies. To use it, run

```sh
conda env create -f environment.yml
conda activate multiarm
```

## Evaluate the pretrained motion planner

In the repo's root, download the pretrained weights and evaluation benchmark
```sh
wget -qO- https://multiarm.cs.columbia.edu/downloads/checkpoints/ours.tar.xz | tar xvfJ -
wget -qO- https://multiarm.cs.columbia.edu/downloads/data/benchmark.tar.xz | tar xvfJ -
```
Then evaluate the pretrained weights on the benchmark with
```sh
python main.py --mode benchmark --tasks_path benchmark/ --load ours/ours.pth --num_processes 1 --gui
```
You can remove `--gui` to run headless, and use more CPU cores with `--num_processes 16`.

To summarize the benchmark results
```sh
python summary.py ours/benchmark_score.pkl
```

## Train a decentralized multi-arm motion planner

In the repo's root, download the training tasks and expert demonstration dataset

```sh
wget -qO- https://multiarm.cs.columbia.edu/downloads/data/tasks.tar.xz | tar xvfJ -
wget -qO- https://multiarm.cs.columbia.edu/downloads/data/expert-demonstrations.tar.xz | tar xvfJ -
```
Then train a decentralized multi-arm motion planner from scratch with 
```sh
python main --configs/default.json --tasks_path tasks/ --expert_waypoints expert-demonstrations/ --num_processes 16
```

## Citation

```
@inproceedings{ha2020multiarm,
  title={Learning a Decentralized Multiarm Motion Planner},
  author={Huy Ha and Jingxi Xu and Shuran Song},
  year={2020},
  booktitle={CoRL},
}
```