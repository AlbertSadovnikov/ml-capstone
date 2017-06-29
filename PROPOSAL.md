# Machine Learning Engineer Nanodegree
## Capstone Proposal
Albert Sadovnikov  
June 13th, 2017


[sample_image]: ./docs/images/sample_image.jpg "Sample aerial image"
[sample_video]: ./docs/images/sample_video.gif "Sample aerial video"


## Traffic analysis of aerial video data

[comment]: <> (https://www.youtube.com/watch?v=_EI7gjQFTdo)
[comment]: <> (https://www.youtube.com/watch?v=1C-BxSt_ub8)


### Domain Background

Automated analytics involving detection, tracking and counting of automobiles
from aerial platform are useful for both commercial and government
purposes. Tracking and counting data can be used to monitor volume and
pattern of traffic as well as volume of parking. It can be more cost effective
than embedding sensors in the road. [1]


### Problem Statement

I have a video of a road roundabout taken from a drone, I would like to detect and track all the cars in it and draw their trajectories.
![sample_video]

The goal of the project is to study the domain and to experiment with latest models publicly available.
Goal for the future would be to detect different vehicle types as well as pedestrians. Also, it would be interesting to estimate object speeds, but this might require finding image to world coordinate transformation.
![sample_image]

### Datasets and Inputs

Model input would be a video file, unfortunately I have only one video sample.

For object detection training I am planning to use a subset of COWC database [1].
Could be that I might need to use some other dataset, which remains to be seen during the development.  

### Solution Statement

Looking at the video the solution could be as follows:
1. Stabilize the video (fix the background).
2. Detect cars on each of the frames.
3. Combine detections through the video to build up trajectories.
4. Create an output video where the cars have bounding boxes drawn over them and also the trajectories are added.

### Benchmark Model

I was not able to find any benchmark model for the project. There are some commercial applications in the domain, but those do not disclose any model details and data.
Ideally, I am aiming at the results obtained in [2].

### Evaluation Metrics

It is hard to make a meaningful evaluation metric in this project, since I don't have the target video annotated.
Annotating this video seems to be quite a tedious task (drawing bounding boxes on each and every frame).

A simple and yet descriptive metric would be to count all the cars appearing in the video and use it as an initial metric.

After there is some model working at least partially - there would be a possibility to get the video annotated automatically and then make some manual fixes.

### Project Design

I am planning to use Python3, OpenCV3 and Keras with Tensorflow backend. Could be that I'll write some routines in C++.

The first step would be to detect cars in standalone frames, then track those through the video using optical flow or some other tracking methods. Tracking can be periodically verified by running a detector on the tracked frames.

I would like to start with Faster R-CNN [3] and see where can I go from there [4][5].

If the results seem promising - I will continue with building up trajectories and image stabilization.

### Potential Problems

It is quite hard to tell if the project is going to be successful or not. The main concern is that there is no data publicly available for this particular task: there are datasets with aerial car images [1] and car images taken "from car" [6].

Quick test with Faster R-CNN trained [7] on Pascal VOC [8] 2007 and 2012 didn't produce any helpful result.

### References

1. [A Large Contextual Dataset for Classification, Detection and Counting of Cars with Deep Learning.](http://gdo-datasci.ucllnl.org/cowc/mundhenk_et_al_eccv_2016.pdf)

2. [Sample video processed by DataFromSky.](https://www.youtube.com/watch?v=XwzbFzqhF1Y&t=7s)

3. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.](https://arxiv.org/abs/1506.01497)

4. [SSD: Single Shot MultiBox Detector.](https://arxiv.org/abs/1512.02325)

5. [You Only Look Once: Unified, Real-Time Object Detection.](https://arxiv.org/abs/1506.02640)

6. [The KITTI Vision Benchmark Suite: Object Detection Evaluation 2012](http://www.cvlibs.net/datasets/kitti/eval_object.php)

7. [An Implementation of Faster RCNN with Study for Region Sampling](https://arxiv.org/pdf/1702.02138.pdf)

8. [The PASCAL Visual Object Classes](http://host.robots.ox.ac.uk/pascal/VOC/)
