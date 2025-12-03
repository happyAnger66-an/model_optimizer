import argparse


from .dataset_utils import _get_dataset_samples

def eval_datasets(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dummy', type=bool, default=False)
    args = parser.parse_args(args[1:])
    print(f'[cli] datasets args {args}')

    _get_dataset_samples(args.name, args.data_dir, 100)