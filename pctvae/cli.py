import sys
import argparse
from pctvae.experiments import (
    nontvae_mnist,
    tvae_Lshort_mnist, 
    tvae_causal_mnist
)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, help='experiment name')

def main():
    args = parser.parse_args()
    module_name = 'pctvae.experiments.{}'.format(args.name)
    experiment = sys.modules[module_name]
    experiment.main()

if __name__ == "__main__":
    main()