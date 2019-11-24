import datetime
import os
import sys
import traceback

from counterfactuals import nn4
from counterfactuals.utilities import get_args_parser


def net_train():
    args = get_args_parser().parse_args()

    # Create results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir + '/' + args.architecture + '-ihdp-predictions' + timestamp + '/'
    os.mkdir(outdir)

    try:

        if args.architecture == 'nn4':
            return nn4.run(args, outdir)
        else:
            return "Architecture not found."

    except Exception as e:
        with open(outdir + 'error.txt', 'w') as error_file:
            error_file.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == "__main__":
    net_train()
