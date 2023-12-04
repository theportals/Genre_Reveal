import matplotlib.pyplot as plt
import pandas as pd
import sys


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: visualize.py <path to csv>")
        sys.exit(-1)
    path = sys.argv[1]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    df = pd.read_csv(path)
    ax.scatter(df.x, df.y, df.z, c=df.c)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")


    plt.show()