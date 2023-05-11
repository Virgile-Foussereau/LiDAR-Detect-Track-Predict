# LiDAR-based object detection, tracking and prediction
# Work done by:
#   - Virgile Foussereau
#   - Jyh-Chwen Ko
# During a Computer Vision course given by Mathieu Brédif at École Polytechnique
"""
Run inference on pcap

Usage:
    $ python detect_pcap_geometry.py --source Ouster-YOLOv5-sample.pcap --metadata-path Ouster-YOLOv5-sample.json

Results are saved to runs2/detect/exp for segmentation and PNG2 for 3D images and predictions
                                                             
"""

import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import time 

from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch


import numpy as np
import cv2
from more_itertools import nth
from sklearn.neighbors import KDTree


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from utils.datasets import IMG_FORMATS, VID_FORMATS
from utils.general import (check_file, increment_path, print_args)


from ouster import client
from ouster import pcap
from contextlib import closing



def run(source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        project=ROOT / 'runs2/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        metadata_path=ROOT / 'example.json'
        ):
    source = str(source)
    is_pcap = source.endswith('.pcap')

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Dataloader
    if is_pcap:
        print('pcap file')
        
        metadata_path = str(metadata_path)

        with open(metadata_path, 'r') as f:
            metadata = client.SensorInfo(f.read())

        fps = int(str(metadata.mode)[-2:])
        print('fps: ', fps)
        width = int(str(metadata.mode)[:4])
        print('width: ', width)
        height = int(str(metadata.prod_line)[5:])
        print('height: ', height)

        pcap_file = pcap.Pcap(source, metadata)
        
        max_range_background_val = np.zeros((height, width), dtype=np.uint8)
        scan_range_field = nth(client.Scans(pcap_file), 45).field(client.ChanField.RANGE)
        scan_range_val = client.destagger(pcap_file.metadata, scan_range_field)
        max_threshold = np.sort(scan_range_val.flatten())[-10]
        print("max_threshold for range: ", max_threshold)
        with closing(client.Scans(pcap_file)) as scans:
            for scan in scans:
                scan_range_field = scan.field(client.ChanField.RANGE)
                scan_range_val = client.destagger(pcap_file.metadata, scan_range_field)
                scan_range_val = np.minimum(scan_range_val, max_threshold)
                max_range_background_val = np.maximum(max_range_background_val, scan_range_val)
        print("Background isolation complete")
        #save max range background in txt file and png file
        #np.savetxt("max_range_background_val.txt", max_range_background_val, fmt='%d')
        #plt.imshow(max_range_background_val, cmap='viridis', resample=False)
        #plt.savefig('max_range_background_val.png')
        

        t0 = time.time()
        pcap_file = pcap.Pcap(source, metadata)

        with closing(client.Scans(pcap_file)) as scans:

            save_pathObjectSegmentation = str(save_dir/"resultsObjectSegmentation.mp4")  # im.jpg
            vid_writerObjectSegmentation = cv2.VideoWriter(save_pathObjectSegmentation, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            centerOfMass_list = []
            k_scan = 0

            for scan in scans:
                k_scan += 1

                range_field = scan.field(client.ChanField.RANGE)
                range_val = client.destagger(pcap_file.metadata, range_field)
                range_val[np.where(range_val==0)] = max_threshold

                xyzlut = client.XYZLut(metadata)
                xyz_destaggered = client.destagger(metadata, xyzlut(scan))

                # detect moving objects
                shape = range_val.shape
                detected_objects = np.zeros(shape, dtype=np.uint8)
                xyz_detected = []
                poi_list = []
                for i in range (shape[0]):
                    for j in range (shape[1]):
                        if range_val[i][j] < max_range_background_val[i,j]-500:
                            poi = (i,j)
                            detected_objects[poi] = 1
                            poi_list.append(poi)
                            xyz_detected.append(xyz_destaggered[i,j])
                #clustering and filtering
                xyz_detected = np.array(xyz_detected)
                xyz_detected_clustered, mask_filtered = clusterAndFilter(xyz_detected)
                n_detected = len(xyz_detected_clustered)
                for i in range(len(xyz_detected)):
                    poi = poi_list[i]
                    detected_objects[poi] = mask_filtered[i]

                #plot detected objects in 3D
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(azim=0, elev=90) #top view
                ax.set_xlim(0, 5)
                ax.set_ylim(-3, 7)
                ax.set_zlim(-1.5, 1.5)
                for i in range(len(xyz_detected_clustered)):
                    xyz_detected_clustered[i] = np.array(xyz_detected_clustered[i])
                    ax.scatter(xyz_detected_clustered[i][:,0], xyz_detected_clustered[i][:,1], xyz_detected_clustered[i][:,2], s=2)


                for i in range(n_detected):
                    centerOfMass = np.mean(xyz_detected_clustered[i], axis=0)
                    #find previous position of the object
                    if k_scan != 0:                 
                        min_dist = 100000
                        min_dist_idx = -1
                        for idx, obj in enumerate(centerOfMass_list):
                            if len(obj) > 0 and len(obj) <= k_scan:
                                dist = np.linalg.norm(obj[-1] - centerOfMass)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_dist_idx = idx
                        if min_dist > 0.5 and n_detected > len(centerOfMass_list):
                            centerOfMass_list.append([centerOfMass])
                            displacement_computed = False
                        else:
                            displacement_computed = True
                            n_frames = min(len(centerOfMass_list[min_dist_idx]), 3)
                            displacement = (centerOfMass - centerOfMass_list[min_dist_idx][-n_frames]) / n_frames
                            centerOfMass_list[min_dist_idx].append(centerOfMass)
                    else:
                        centerOfMass_list.append([centerOfMass])
                        displacement_computed = False
                    ax.scatter(centerOfMass[0], centerOfMass[1], centerOfMass[2], c='k', marker='x')
                    if displacement_computed:
                        arrow_vec = displacement/np.linalg.norm(displacement)
                        ax.arrow3D(centerOfMass[0], centerOfMass[1], centerOfMass[2],
                                    arrow_vec[0], arrow_vec[1], 0, 
                                    mutation_scale=10,
                                    ec ='red',
                                    fc='red',
                                    alpha=1)
                plt.savefig('PNG2/arrow'+str(k_scan)+'.png')
                plt.close()
                colored = np.zeros((height, width, 3), dtype=np.uint8)
                labels = {} #map labels to [0,1,2,...,n_detected-1] 
                colors = plt.cm.rainbow(np.linspace(0, 1, n_detected))
                #remove column 4 from color map
                colors = np.delete(colors, 3, 1)
                for i in range (shape[0]):
                    for j in range (shape[1]):
                        label_detected = detected_objects[i][j]
                        if label_detected != 0:
                            if label_detected not in labels:
                                labels[label_detected] = len(labels)
                            colored[i,j] = colors[labels[label_detected]]*255
                vid_writerObjectSegmentation.write(colored)
            print('Object segmentation done')
            t1 = time.time()
            print('Time for detection: ', t1-t0)
            print('Video duration: ', k_scan/fps)   
            vid_writerObjectSegmentation.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--project', default=ROOT / 'runs2/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--metadata-path', type=str, default=ROOT / 'example.json', help='metadata path')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt  

def outlierRemoval3d(points, k=2):
    """
    Remove outliers from a point cloud using the 3D distance to the kth nearest neighbor.
    :param points: Nx3 numpy array of 3D points
    :param k: number of nearest neighbors to use for outlier removal
    :return: Nx3 numpy array of inlier 3D points and mask of the outliers
    """
    tree = KDTree(points, leaf_size=40)
    dist, _ = tree.query(points, k=k+1)
    dist = dist[:, k]
    thresh = 3 * np.mean(dist)
    mask = np.where(dist > thresh, 1, 0)
    return points[dist < thresh], mask

def clusterAndFilter(points, r=0.07, min_points=3):
    """
    Cluster and filter a point cloud using DBSCAN.
    :param points: Nx3 numpy array of 3D points
    :param r: DBSCAN radius
    :param min_points: DBSCAN minimum number of points
    :return: Nx3 numpy array of clustered and filtered 3D points and mask of the outliers
    """
    db = DBSCAN(eps=r, min_samples=min_points).fit(points)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    clusters = []
    for i in range(n_clusters_):
        clusters.append([])
    for i in range(len(labels)):
        if labels[i] != -1:
            clusters[labels[i]].append(points[i])
    #remove clusters with less than 500 points
    for i in range(len(clusters)):
        if len(clusters[i]) < 500:
            clusters[i] = []
            labels = np.where(labels == i, -1, labels)
    clusters = [x for x in clusters if x != []]
    mask_clusters = np.where(labels == -1, 0, labels+1)
    return clusters, mask_clusters

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    setattr(Axes3D, 'arrow3D', _arrow3D)
    opt = parse_opt()
    main(opt)