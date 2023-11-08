import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure()
ax = plt.axes(projection='3d')
df = pd.read_csv("output.csv")
ax.scatter(df.x, df.y, df.z, c=df.c)
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")


plt.show()