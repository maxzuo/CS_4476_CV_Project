<center>
    Team members: Mitchell Stasko, Michael Verges, Max Zuo
</center>

## Project Proposal

In the last several years, advances in computer vision and machine learning have led to significant improvements in object detection and instance segmentation algorithms. State-of-the-art deep learning models such as Mask R-CNN and even more recent approaches have improved accuracy while models like Faster R-CNN, MobileNet, and YOLO have pushed the boundaries for speed.


As we continue to improve the capabilities of our computer vision systems, a logical next step from object detection is object tracking, an inherent capability in human and animal vision. Short-term and long-term object tracking remain open and challenging problems, as they require both sufficient accuracy and speed to be able to process video data in a reasonable amount of time. Object tracking algorithms may also require an understanding of object permanence in order to correctly handle scenarios with momentary object occlusion.


In this project, we will be exploring the effectiveness of a number of ROI object tracking algorithms, including KCF, CSRT, TLD, and others. Performance will be measured on both frames per second the algorithm can achieve as well as the accuracy of the algorithms measured using mean Average Precision (mAP) over the labeled dataset.


A success for this project would be having comparable data to say “object tracking algorithm x is the best because of these reasons determined through experimentation” or “all of these tracking algorithms are comparable in general but for specialties such as A, B, or C, algorithms X, Y, and Z excel respectively and here’s why with experimental data:”. Of course, we don’t currently know what the results would be but it’s an exciting prospect to take many people’s different ideas on how to solve the same problem and see just how the performance of these solutions differ and how each implementation could possibly lend itself to a different industry.


## References:
[TrackingNet](https://tracking-net.org/)

[CSRT](https://arxiv.org/pdf/1611.08461)

[KCF](https://arxiv.org/abs/1404.7584)

[TLD](https://ieeexplore.ieee.org/document/6104061) and [e-TLD](https://arxiv.org/abs/2009.00855)