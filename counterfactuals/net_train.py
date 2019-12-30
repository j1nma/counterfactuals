import datetime
import os
import sys
import traceback
from pathlib import Path

from counterfactuals import nn4, cnn, nn4_vb
from counterfactuals.utilities import get_nn_args_parser


def net_train(config_file):
    args = get_nn_args_parser().parse_args(['@' + config_file])

    # Create results directory
    outdir_path = Path(args.outdir)
    if not outdir_path.is_dir():
        os.mkdir(args.outdir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir + '/' + timestamp + '/'
    os.mkdir(outdir)

    try:

        if args.architecture == 'nn4':
            return nn4.run(args, outdir)
        elif args.architecture == 'cnn':
            return cnn.run(args, outdir)
        elif args.architecture == 'nn4_vb':
            return nn4_vb.run(args, outdir)
        else:
            return "Architecture not found."

    except Exception as e:
        with open(outdir + 'error.txt', 'w') as error_file:
            error_file.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == "__main__":
    net_train(config_file=sys.argv[1])
