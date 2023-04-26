
import sys
sys.path.append("..")
from third_party import utils
from third_party.run_eval import run_eval

def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    run_eval(config)

if __name__ == '__main__':
    main()
