def main():
    from . import launcher

    launcher.launch()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()