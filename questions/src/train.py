"""Training and evaluation. Usage: python src/train.py --model {temporal,fixed,adaptive,hybrid}"""
import argparse
from utils import set_seed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["temporal", "fixed", "adaptive", "hybrid"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    raise NotImplementedError

if __name__ == "__main__":
    main()
