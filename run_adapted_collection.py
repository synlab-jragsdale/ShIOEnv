import argparse
import json

from ShIOEnv.agent import ShIOAgent


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, required=False, default='config.json', help='Path to config file')
    parser.add_argument('-l', '--log', type=str, required=True, help='logging runprefix (inner directories in data/)')
    parser.add_argument('-d', '--debug', action='store_true', help='debug switch')
    parser.add_argument('-w', '--workers', type=int, default=30, required=False, help='number of workers to work in parallel (may cause misallignment with update_steps as episodes finish)')
    parser.add_argument('-n', '--num', type=int, default=1000, required=False, help='Size of dataset in thousands')

    parser.add_argument('-r', '--read', type=str, required=False, default='maps/raw_nl2cmd_seqs.json', help='path to input-only file')
    parser.add_argument('--docker', action='store_true', help="Use Docker executor")

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    config["runner"]["runprefix"] = args.log

    with open(args.read, 'r') as f:
        cmds = json.load(f)

    config['runner']['debug'] = args.debug
    config['runner']['n_workers'] = args.workers if not args.debug else 1

    config['env']['exec_method'] = 'docker' if args.docker else 'firecracker'
    config['env']['local_horizon'] = 30
    config['env']['get_final_score'] = False

    config['dataset']['dataset_size'] = args.num
    config['dataset']['dataset_store_every'] = 1000

    handler = ShIOAgent(config=config)
    handler.run_inputonly_creation(cmds)

if __name__ == "__main__":
    main()