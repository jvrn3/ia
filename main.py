import argparse
from training import train_data


def main(data):
    try:
        train_data(data)
    except FileNotFoundError:
        print("Dataset não encontrado")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trabalho Perceptron")
    parser.add_argument("file_name", help="dataset name")
    args = parser.parse_args()
    main(args.file_name)
