"""Run frozen Algorithm4 on a precomputed feature CSV."""
import argparse
import pandas as pd
from .core import predict

def main():
    """Read feature CSV, apply frozen policy, and save research predictions."""
    p=argparse.ArgumentParser(description='Algorithm4 capital-impairment research inference')
    p.add_argument('input');p.add_argument('output');p.add_argument('--horizon',type=int,choices=[12,24],default=12)
    a=p.parse_args()
    frame=pd.read_csv(a.input,dtype={'corp_code':str})
    predict(frame,a.horizon).to_csv(a.output,index=False)
if __name__=='__main__':main()
