import datetime as dt
import numpy as np
from pred_NBA import run

def main():
    today = dt.date.today()
    from sys import argv
    script, method = argv
    run(2015, method).to_csv(
        'output/predictions_{YEAR}_{MONTH}_{DAY}.csv'.format(
            YEAR=today.year, MONTH=today.month, DAY=today.day
        )
    )

if __name__=='__main__':
    main()
