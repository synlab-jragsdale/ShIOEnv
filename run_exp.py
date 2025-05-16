import argparse
import json

from ShIOEnv.agent import ShIOAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=False, default='ShIOEnv/policy_config.json', help='Path to config file')

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)

    handler = ShIOAgent(config=config)
    handler.run_periodic()

if __name__ == "__main__":
    main()
