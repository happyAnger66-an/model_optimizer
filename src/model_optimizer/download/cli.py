import argparse
def download_cli(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args(args[1:])

    model_name = args.model_name
    model_path = args.model_path
    output_dir = args.output_dir

    from openpi.shared import download

    checkpoint_dir = download.maybe_download(model_path)
    print(f'download model {model_name} to {checkpoint_dir}')