import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from collections import Counter

color_enum = ["r", "g", "b", "c", "m", "y", "k", "w"]

dir_name = "/home/xt/Documents/data/surfelwarp/build/apps/surfelwarp_kinectdk/temp/"
#dir_name = "/home/xt/Documents/data/surfelwarp/build/apps/surfelwarp_app/temp/"

for i in range(201, 300, 7):
#for i in range(250, 285, 7):
    data = np.loadtxt(dir_name + "se3_frame_%06d.log" % i)
    se3, coord = data[:, :8], data[:, 8:12]


    db = DBSCAN(eps=0.01, min_samples=8)
    db = db.fit(se3)
    cnt = Counter(db.labels_)
    print("%d: %s" % (i, str(cnt)))

    if(1):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.view_init(elev=20, azim=-90)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        colors = [color_enum[x % len(color_enum)] for x in db.labels_]
        for j in range(len(se3)):
            #ax.quiver(coord[j, 0], coord[j, 1], coord[j, 2], 1, 1, 1, length=0.01, color=colors[j])
            # for view convenient
            ax.quiver(coord[j, 0], coord[j, 2], -coord[j, 1], 1, 1, 1, length=0.01, color=colors[j])
        plt.show()
