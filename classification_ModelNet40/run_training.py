import yaml
import argparse
import os
import subprocess
import itertools
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to base config file.')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_command(config):
    cmd = ['python', 'main.py']
    for key, value in config.items():
        if value in [None, "", [], {}]:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        elif isinstance(value, list):
            cmd.append(f'--{key}')
            cmd.extend([str(v) for v in value])
        else:
            cmd.append(f'--{key}')
            cmd.append(str(value))
    return ' '.join(cmd)

def generate_and_submit_jobs(base_config):
    receptive_fields = ['knn', 'radius']
    seeds = [42, 420, 4200]
    samplings = ['uniform', 'gradient', 'split']

    sweep_combinations = list(itertools.product(receptive_fields, seeds, samplings))

    for rf, seed, sampling in sweep_combinations:
        config = copy.deepcopy(base_config)
        config['receptive_field'] = rf
        config['seed'] = seed
        config['sampling'] = sampling
        config['msg'] = f"sweep_rf-{rf}_seed-{seed}_samp-{sampling}"

        command = build_command(config)

        sbatch_path = os.path.join("sbatch_folder", f"run_train.sbatch")

        print(f"Submitting job: {sbatch_path}")
        subprocess.run(['sbatch', sbatch_path])

def main():
    args = parse_args()
    base_config = load_config(args.config)
    generate_and_submit_jobs(base_config)

if __name__ == '__main__':
    main()
