import argparse

def solicit_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', help='iterations', type=int, default=100000)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=512)
    parser.add_argument('--num_blocks', help='repeat blocks', type=int, default=2)

    parser.add_argument('--dev_batch_size', help='dev_batch_size', type=int, default=1024)
    parser.add_argument('--iter2report', help='iterations to report', type=int, default=1000)
    parser.add_argument('--version', help='version', type=str, default='3')
    parser.add_argument('--init_lr', help='initialized learning rate', type=float, default=0.0001)
    return parser.parse_args()