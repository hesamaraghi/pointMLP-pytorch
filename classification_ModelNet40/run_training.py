import yaml
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file.')
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
    return cmd

def main():
    args = parse_args()
    config = load_config(args.config)
    command = build_command(config)
    print("Running command:")
    print(" ".join(command))
    subprocess.run(command)

if __name__ == '__main__':
    main()
