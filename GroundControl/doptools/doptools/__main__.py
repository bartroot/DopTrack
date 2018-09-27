import argparse

from .ftp import download_eopp, download_eopc04, download_tai_utc


parser = argparse.ArgumentParser(description='DopTrack tools')
parser.add_argument('-a',
                    '--analyze',
                    action='store',
                    help='Analyze full dataset')
parser.add_argument('-d',
                    '--download',
                    action='store_true',
                    default=False,
                    help='Download .eopp files')


def main():
    args = parser.parse_args()

    if args.download:
        download_eopp()
        download_eopc04()
        download_tai_utc()


if __name__ == '__main__':
    main()
