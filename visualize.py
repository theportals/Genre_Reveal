import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns

fig = plt.figure()
ax = plt.axes(projection='3d')
df = pd.read_csv("output.csv")
# sns.scatterplot(x=df.x, y=df.y, z=df.z, hue=df.c, palette=sns.color_palette("hls", n_colors=10))
ax.scatter(df.x, df.y, df.z, c=df.c)
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")


plt.show()