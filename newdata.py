import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob
from plot import Plot


def main():
    local = os.getcwd()
    path = r"C:\\Users\\Kazuki\\thesis\\newdata"
    os.chdir(path)
    files = glob("*.csv")

    data = []
    newdata = pd.DataFrame(
        index=[], columns=['t', 'positive', 'negative', 'neutral', 'total'])
    for f in files:
        original = pd.read_csv(f, encoding='utf-8')
        original = original.drop(original.columns[[0, 6, 7, 8]], axis=1)
        data.append(original)
    for df in data:
        newdata = newdata.add(df, fill_value=0)
    newdata['t'] = newdata.index + 1
    newdata['positive(%)'] = [i / j * 100 if j != 0 else None for i,
                              j in zip(newdata['positive'], newdata['total'])]
    newdata['negative(%)'] = [i / j * 100 if j != 0 else None for i,
                              j in zip(newdata['negative'], newdata['total'])]
    newdata['neutral(%)'] = [i / j * 100 if j != 0 else None for i,
                             j in zip(newdata['neutral'], newdata['total'])]
    newdata.to_csv("c:/Users/Kazuki/thesis/new_data.csv")

    pl = Plot()
    k = 0
    x = [newdata['positive'][k], newdata['negative'][k], newdata['neutral'][k]]
    pl.dirichlet(x)

    fig, ax = plt.subplots()
    ax.set_xlabel("t")
    ax.set_ylabel("Percentage")
    ax.set_title("time and sentiment")
    ax.grid()
    ax.set_xlim([0, 50])
    ax.plot(newdata['t'], newdata['positive(%)'], label="Positive")
    ax.plot(newdata['t'], newdata['negative(%)'], label="Negative")
    ax.plot(newdata['t'], newdata['neutral(%)'], label="Neutral")
    ax.legend(loc=0)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
