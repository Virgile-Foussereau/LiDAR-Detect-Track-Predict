#  LiDAR-based pedestrian detection, tracking and prediction based on YOLOv5 architecture
# Work done by:
#   - Virgile Foussereau
#   - Jyh-Chwen Ko
# During a Computer Vision course given by Mathieu Brédif at École Polytechnique
"""
Run inference on pcap

Usage:
    $ python detect_PCAP_learning.py --class 0 --weights best.pt --conf-thres=0.4 --source Ouster-YOLOv5-sample.pcap --metadata-path Ouster-YOLOv5-sample.json  --view-img
                                                             
"""

import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import pickle

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from more_itertools import nth
from sklearn.neighbors import KDTree

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadNumpy
from utils.general import (LOGGER, check_file, check_img_size, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from ouster import client
from ouster import pcap
from contextlib import closing
import logging

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        social_distance=False,
        metadata_path=ROOT / 'example.json'
        ):
    source = str(source)
    is_pcap = source.endswith('.pcap')

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

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
        
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

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
        #Uncomment to save max range background in txt file and png file
        # np.savetxt("max_range_background_val.txt", max_range_background_val, fmt='%d')
        # plt.imshow(max_range_background_val, cmap='viridis', resample=False)
        # plt.savefig('max_range_background_val.png')


        pcap_file = pcap.Pcap(source, metadata)

        with closing(client.Scans(pcap_file)) as scans:

            save_path = str(save_dir/"results.mp4")  # im.jpg
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            bs = 1 # batch_size
            save_pathHumanSegmentation = str(save_dir/"resultsHumanSegmentation.mp4")  # im.jpg
            vid_writerHumanSegmentation = cv2.VideoWriter(save_pathHumanSegmentation, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            centerOfMass_list = []
            k_scan = 0

            for scan in scans:
                k_scan += 1
                ref_field = scan.field(client.ChanField.REFLECTIVITY)
                ref_val = client.destagger(pcap_file.metadata, ref_field)
                #ref_img = (ref_val / np.max(ref_val) * 255).astype(np.uint8)
                ref_img = ref_val.astype(np.uint8)

                range_field = scan.field(client.ChanField.RANGE)
                range_val = client.destagger(pcap_file.metadata, range_field)
                #range_img = (range_val / np.max(range_val) * 255).astype(np.uint8)
                #range_img = range_val

                combined_img = np.dstack((ref_img, ref_img, ref_img))

                xyzlut = client.XYZLut(metadata)
                xyz_destaggered = client.destagger(metadata, xyzlut(scan))

                #run inference
                dataset = LoadNumpy(numpy=combined_img, path="", img_size=imgsz, stride=stride, auto=pt and not jit)

                if pt and device.type != 'cpu':
                    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
                dt, seen = [0.0, 0.0, 0.0], 0

                for path, im, im0s, vid_cap, s in dataset:
                    t1 = time_sync()
                    im = torch.from_numpy(im).to(device)
                    im = im.half() if half else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1

                # Inference
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2

                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    #save_path = str(save_dir / p.name)  # im.jpg
                    #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    txt_path = str(save_dir / 'labels' / p.stem) 
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                        
                        poi_list = []
                        xyz_list = []
                        xyxy_list = []
                        range_list = []
                        detectedHuman = np.zeros(range_val.shape)
                        n_detected = 0

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            n_detected += 1
                            xyxy_list.append(xyxy)

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                x1 = int(xyxy[0])
                                y1 = int(xyxy[1])
                                x2 = int(xyxy[2])
                                y2 = int(xyxy[3])

                                range_roi = range_val[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] #whole box
                                range_roi[np.where(range_roi==0)] = max_threshold #add a big number to zero range                      
                        
                                min_range = np.min(range_roi) 
                                range_list.append(min_range)
                                
                                multiple_poi_list = []
                                multiple_xyz_list = []
                                lines, columns = range_roi.shape
                                for i in range(lines):
                                    for j in range(columns):
                                        poi_x = j + x1
                                        poi_y = i + y1
                                        poi = (poi_y, poi_x) #(y,x) in global
                                        if range_val[poi] <= max_range_background_val[poi]-500: 
                                            multiple_poi_list.append(poi)
                                            xyz_val = xyz_destaggered[poi]
                                            multiple_xyz_list.append(xyz_val)
                                            detectedHuman[poi] = 1
                                
                                # transform scan data to 3d points
                                multiple_xyz_list = np.array(multiple_xyz_list)

                                # Remove outliers
                                multiple_xyz_list_reduced, mask_outliers = outlierRemoval3d(multiple_xyz_list, k=4)
                                for i in range(len(multiple_poi_list)):
                                    if mask_outliers[i]:
                                        detectedHuman[multiple_poi_list[i]] = 0
                                


                                # Compute centroid
                                centerOfMass = np.mean(multiple_xyz_list_reduced, axis=0)
                                closestPointToCenterOfMass_idx = np.argmin(np.linalg.norm(multiple_xyz_list - centerOfMass, axis=1))
                                closestPointToCenterOfMass = multiple_xyz_list[closestPointToCenterOfMass_idx]
                                closestPointToCenterOfMass_poi = multiple_poi_list[closestPointToCenterOfMass_idx]

                                #find previous position of the person
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

                                
                                

                                

                                #plt.imshow(detectedHuman, cmap='gray', resample=False)
                                #plt.savefig('detectedHuman.png')

                                #uncomment to plot 3d points of detected human
                                colors_detected = ['y', 'g', 'b', 'k']
                                color_choice = colors_detected[n_detected % len(colors_detected)]
                                ax.scatter(multiple_xyz_list_reduced[:,0], multiple_xyz_list_reduced[:,1], multiple_xyz_list_reduced[:,2], c=color_choice, marker='o', alpha=0.3)
                                ax.scatter(centerOfMass[0], centerOfMass[1], centerOfMass[2], c='k', marker='x')
                                if displacement_computed:
                                    ax.arrow3D(centerOfMass[0], centerOfMass[1], centerOfMass[2],
                                                displacement[0]*30, displacement[1]*30, 0, #prediction on next 30 frames
                                                mutation_scale=10,
                                                ec ='red',
                                                fc='red',
                                                alpha=1)
                                ax.scatter(closestPointToCenterOfMass[0], closestPointToCenterOfMass[1], closestPointToCenterOfMass[2], c='g', marker='o')
                                ax.view_init(azim=0, elev=90) #top view
                                ax.set_xlim(0, 5)
                                ax.set_ylim(-3, 7)
                                ax.set_zlim(-1.5, 1.5)
                                

                                poi_list.append(closestPointToCenterOfMass_poi)

                                if social_distance == False:
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                        
                                xyz_val = xyz_destaggered[poi]
                                xyz_list.append(xyz_val)
                    
                                if save_crop:
                                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            

                        plt.savefig('PNG/arrow'+str(k_scan)+'.png')
                        plt.close()
                        imHumanSegmentation = (detectedHuman * 255).astype(np.uint8)
                        imSeg_stack = np.dstack((imHumanSegmentation, imHumanSegmentation, imHumanSegmentation))

                        import csv
                        if save_txt:
                            csv_file = open(txt_path + '.csv', 'a', newline='')
                            writer = csv.writer(csv_file)
                        

                        if social_distance == True:

                            if len(poi_list) < 2: 
                                print('just 1 object')
                                if save_txt:  # Write to file
                                    writer.writerow([1, 0, 0])
                            else:
                                xyz_1 = xyz_list[0]
                                xyz_2 = xyz_list[1]

                                import math
                                dist = math.sqrt((xyz_1[0] - xyz_2[0])**2 + (xyz_1[1] - xyz_2[1])**2 + (xyz_1[2] - xyz_2[2])**2)

                                annotator.display_distance(xyxy_list[0], poi_list[0], label, dist, color=colors(c, True))
                                annotator.display_distance(xyxy_list[1], poi_list[1], label, dist, color=colors(c, True))

                                if save_txt:  # Write to file
                                    writer.writerow([2, dist, 1 if dist < 1.8 else 0])
                    
                    # Print time (inference-only)
                    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                    # Stream results
                    im0 = annotator.result()
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond


                    vid_writer.write(im0)
                    vid_writerHumanSegmentation.write(imSeg_stack)  

                    


            #vid_writer.release()
            vid_writerHumanSegmentation.release()
                        

            # Print results
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if update:
                strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--social-distance', action='store_true', help='calculate distance between two people')
    parser.add_argument('--metadata-path', type=str, default=ROOT / 'example.json', help='metadata path')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
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
    thresh = np.mean(dist)*1.1
    print("thresh", thresh)
    print("mean", np.mean(dist))
    mask = np.where(dist > thresh, 1, 0)
    return points[dist < thresh], mask

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
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    setattr(Axes3D, 'arrow3D', _arrow3D)
    opt = parse_opt()
    main(opt)