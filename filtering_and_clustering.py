import numpy as np
import csv
import matplotlib.pyplot as plt
import os

if not os.path.exists("clustersPNG"):
    os.makedirs("clustersPNG")




# Read the csv file if it exists
if os.path.isfile('xyz_detected.csv'):
    with open('xyz_detected.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = np.array(data)
        data = data.astype(float)
else:
    print("No xyz_detected.csv file found")
    exit()

#remove rows with 0
data = data[abs(data[:,0]) > 0.0001]
print("Number of detected points: ", len(data))


#compute matrix of distances between points
distances = np.zeros((len(data), len(data)))
for i in range(len(data)):
    for j in range(len(data)):
        distances[i][j] = np.linalg.norm(data[i] - data[j])

print("distances computed")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#plot all points in 3D
colors = plt.cm.rainbow(np.linspace(0, 1, len(data)))
ax.scatter(data[:,0], data[:,1], data[:,2], c=colors, s=1)
fig.tight_layout()
plt.savefig("clustersPNG/step" + str(0) + ".png")

#union find algorithm
parent = [i for i in range(len(data))]
rank = [0 for i in range(len(data))]
def find(x):
    if x != parent[x]:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    x = find(x)
    y = find(y)
    if x == y:
        return
    if rank[x] > rank[y]:
        parent[y] = x
        #update color in 3D plot
        colors[y] = colors[x]
    else:
        parent[x] = y
        #update color in 3D plot
        colors[x] = colors[y]
        if rank[x] == rank[y]:
            rank[y] += 1

#find clusters and plot along the way
for i in range(len(data)):
    for j in range(i+1, len(data)):
        if distances[i][j] < 0.08:
            union(i, j)
    if i % 100 == 0:
        print("step", i)
        ax.clear()
        ax.scatter(data[:,0], data[:,1], data[:,2], c=colors, s=1)
        fig.tight_layout()
        plt.savefig("clustersPNG/step" + str(i) + ".png")

ax.clear()
ax.scatter(data[:,0], data[:,1], data[:,2], c=colors, s=1)
fig.tight_layout()
plt.savefig("clustersPNG/stepFinal.png")
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()
z_lim = ax.get_zlim()
ax.clear()
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.set_zlim(z_lim)

#delete clusters with less than 500 points
clusters = {}
for i in range(len(data)):
    if find(i) not in clusters:
        clusters[find(i)] = []
    clusters[find(i)].append(data[i])

for key in list(clusters.keys()):
    if len(clusters[key]) < 500:
        del clusters[key]
    else:
        ax.scatter(np.array(clusters[key])[:,0], np.array(clusters[key])[:,1], np.array(clusters[key])[:,2], s=1, c=colors[key])

fig.tight_layout()
plt.savefig("clustersPNG/stepFinalClusters.png")

print("Number of clusters: ", len(clusters))
print("points per cluster: ", [len(clusters[key]) for key in clusters])