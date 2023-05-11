# LiDAR-based object detection, tracking and prediction

This project aims at detecting, tracking and predicting movement of objects, especially pedestrian, using LiDAR data. Please learn more about it in the [report](Report_LiDAR_project.pdf).

## Installation

All required packages are listed in `requirements.txt`. We recommend using a conda virtual environment to install them using the following command.

```bash
conda create --name <env> --file requirements.txt
```

## Usage

1. The learning-based detection can be run using the following command:

```python
python detect_PCAP_learning.py --class 0 --weights best.pt --conf-thres=0.4 --source Ouster-YOLOv5-sample.pcap --metadata-path Ouster-YOLOv5-sample.json  --view-img
```

2. The geometry-based detection can be run using the following command:

```python
python detect_pcap_geometry.py --source Ouster-YOLOv5-sample.pcap --metadata-path Ouster-YOLOv5-sample.json
```

3. The results are saved in `runs` (resp. `runs2`) and `PNG` (resp. `PNG2`) for the learning-based (resp. geometry-based) approach. To animate the PGN files into a video, run:

```python
python animate_png.py
```

The results are saved in `animations`.

## References

This project was done by Virgile Foussereau and Jyh-Chwen Ko during a Computer Vision course given by Mathieu Brédif at École Polytechnique. The learning-based method is based upon a [work](https://github.com/fisher-jianyu-shi/yolov5_Ouster-lidar-example) on social distancing done by Fisher Jianyu.