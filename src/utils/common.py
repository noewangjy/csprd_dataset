import timeit, datetime
from functools import wraps

import pandas as pd
import seaborn as sns


def log_step(funct):
    @wraps(funct)
    def wrapper(*args, **kwargs):
        tic = timeit.default_timer()
        result = funct(*args, **kwargs)
        time_taken = datetime.timedelta(seconds=timeit.default_timer() - tic)
        print(f"Just ran '{funct.__name__}' function. Took: {time_taken}")
        return result
    return wrapper


def plot_bm25_heatmap(filepath: str = "../output/zeroshot/gridsearch/bm25_results.csv"):
    df = pd.read_csv(filepath)
    df = df.pivot_table(values='recall@100', index='k1', columns='b')[::-1] *100
    plot = sns.heatmap(df, annot=True, cmap="YlOrBr", fmt='.1f', center=40.0)
    plot.get_figure().savefig("bm25_heatmap.png")
