import sys
sys.path.append("..")
from third_party import utils
from third_party.run_train import run

def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    print(config)

    # EMA object to track loss and discrminator predictions
    ema_losses = utils.ema_losses(start_itr=1000)
    run(config, ema_losses)

if __name__ == '__main__':
    main()
