import argparse
import json

from ShIOEnv.agent import ShIOAgent


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, required=False, default='config.json', help='Path to config file')
    parser.add_argument('-l', '--log', type=str, required=True, help='logging runprefix (inner directories in data/')
    parser.add_argument('-d', '--debug', action='store_true', help='debug switch')
    parser.add_argument('-w', '--workers', type=int, default=30, required=False, help='number of workers to work in parallel')
    parser.add_argument('-n', '--num', type=int, default=1000, required=False, help='Dataset size')

    parser.add_argument('-m', '--max', type=int, default=10, required=False, help='max argument horizon')
    parser.add_argument('-r', '--repeats', type=int, default=7, required=False, help='1/r chance to end Kleene productions')
    parser.add_argument('-a', '--approx', type=int, default=32, required=False, help='number of permutations to evaluate')
    parser.add_argument('-u', '--util', type=str, required=False, default="maps/utils.cmd", help='Path to util file')
    parser.add_argument('--all', action='store_true', help='Use all permutations to calculate irreducibility error')
    parser.add_argument('--no-score', action='store_true', help='Do not execute sub-inputs to calculate irreducibility')
    parser.add_argument('--nomask', action='store_true', help="Allow any production to be used for argument construction")
    parser.add_argument('--docker', action='store_true', help="Use Docker executor")

    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    config["runner"]["runprefix"] = args.log

    config['runner']['debug'] = args.debug
    config['runner']['n_workers'] = args.workers if not args.debug else 1

    config['runner']['grammar_mask'] = not args.nomask
    config['runner']['early_stop_repeated'] = True
    config['runner']['random_arg_weight'] = args.repeats
    config['env']['get_final_score'] = not args.no_score

    config['env']['local_horizon'] = args.max
    config['env']['contrib_samples'] = args.approx
    config['env']['sample_all_combs'] = args.all
    config['env']['exec_method'] = 'docker' if args.docker else 'firecracker'

    config['dataset']['dataset_size'] = args.num
    config['dataset']['dataset_store_every'] = 1000

    with open(args.util, 'r') as f:
        config['runner']['test_cmd'] = ([_.strip() for _ in f.readlines()])

    handler = ShIOAgent(config=config)
    handler.run_dataset_creation()

if __name__ == "__main__":
    main()