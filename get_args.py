import argparse

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', default  = 'cifar')

    # Neural Network setting
    parser.add_argument('-cout', type=int, default  = 12)
    parser.add_argument('-cfeat', type=int, default  = 256)

    # The transmitter setting
    parser.add_argument('-distribute', default  = False)
    parser.add_argument('-res', default  = True)
    parser.add_argument('-diversity', default  = True)
    parser.add_argument('-adapt', default  = True)
    parser.add_argument('-Nt',  default  = 2)
    parser.add_argument('-P1',  default  = 10.0)
    parser.add_argument('-P2',  default  = 10.0)
    parser.add_argument('-P1_rng',  default  = 4.0)
    parser.add_argument('-P2_rng',  default  = 4.0)

    # The receiver setting
    parser.add_argument('-Nr',  default  = 2)

    # training setting
    parser.add_argument('-epoch', type=int, default  = 400)
    parser.add_argument('-lr', type=float, default  = 1e-4)
    parser.add_argument('-train_patience', type=int, default  = 12)
    parser.add_argument('-train_batch_size', type=int, default  = 32)

    parser.add_argument('-val_batch_size', type=int, default  = 32)
    parser.add_argument('-resume', default  = False)
    parser.add_argument('-path', default  = 'models/')

    args = parser.parse_args()

    return args
