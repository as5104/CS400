"""Dedicated process for a manually started dashboard job."""
import argparse
from .local_jobs import execute_job

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--job',required=True)
    args=parser.parse_args();execute_job(args.root,args.job)

if __name__=='__main__':main()
