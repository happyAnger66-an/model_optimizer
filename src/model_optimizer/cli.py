def main():
    from . import launcher

    launcher.launch()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    from utils.log import setup_logging
    setup_logging()

    freeze_support()
    main()