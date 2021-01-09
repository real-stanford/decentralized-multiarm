# Running the 6 DOF Bin Pick and Place Demo

In the `demo/` directory, download the benchmark tasks to the `tasks/` directory
```
wget -qO- https://multiarm.cs.columbia.edu/downloads/data/benchmark.tar.xz | tar xvfJ -
mv benchmark tasks/
```

To run the demo for 500 experiments
```sh
python demo.py
```

This will create a results folder, where the results csv file will be saved, and a simulation folder, where the simulation pickles will be saved.

To evaluate the results
```sh
python evaluate_results.py --result_dir path/to/results/dir
```
where `path/to/results/dir` looks something like `results_2021-01-01_18-06-35`. The output should look something like
```json
{
    "num_exps": 500,
    "num_valid_exps": 298,
    "num_success": 232,
    "avg_steps": 6206.353448275862,
    "num_plane_collision": 31,
    "num_robot_collision": 35,
    "num_timeout": 0,
    "success_rate": 0.7785234899328859
}
```

To visualize the simulations, install the [PyBullet Blender Plugin](https://github.com/huy-ha/pybullet-blender-recorder), then import the desired simulation pickle file in the simulation folder from Blender's GUI.