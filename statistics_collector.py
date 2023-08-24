import csv

import numpy as np


def main():
    ratios = np.zeros(30)
    for i in range(30):
        with open(f"timings/{i + 1}.csv", 'r', newline='') as f:
            reader = csv.DictReader(f, delimiter=',')
            stats = next(reader)
            ratios[i] = float(stats['TorchGNN']) / float(stats['PyTorch'])
    with open("ratio_summary.csv", 'w') as f:
        f.write(f"Mean, Standard deviation\n")
        f.write(f"{np.mean(ratios)},{np.std(ratios)}")


if __name__ == '__main__':
    main()
