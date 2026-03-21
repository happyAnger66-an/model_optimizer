def main():
    from . import launcher

    launcher.launch()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    from model_optimizer.utils.log import setup_logging

    setup_logging()
    freeze_support()
    main()