<style>
.tablelines table, .tablelines td, .tablelines th {
    border: 1px solid black;
}
.result {
    width: 30%;
}
.teaser {
    display: block;
    width: 45%;
    margin: auto;
}
</style>

<center>
    Team members: Mitchell Stasko, Michael Verges, Max Zuo
</center>

## Project Update #1

## Abstract:
Not only is it important to understand what an object is, it is useful to know where the object is. We wanted to use computer vision to help the process of tracking objects by examining popular object tracking techniques and making apples to apples comparisons. After studying the popular CSRT object tracking procedure, we implemented opencvs implementation of CSRT and ran it on a subset of the TrackingNet dataset, the largest, free object tracking dataset, to evaluate its implementation. We did the same for KCF, implementing opencvs implementation and running it on the TrackingNet dataset. We received informative results of mildly inaccurate object tracking from CSRT and moderate to largely inaccurate object tracking from KCF on the massive amount of data we ran the tracking on that took ~24 hours on one thread to fully execute.


![CSRT Example](img/sample5.gif){: .teaser}

![KCF Example](img/kcf8.gif){: .teaser}


## Introduction:
Object tracking is an increasingly important part of the modern world as technology becomes more ubiquitous and cameras and computers see more of our every move. Object tracking like such has applications in the field of surveillance, traffic flow analysis, self driving vehicles, crowd counting, audience flow analysis, and many more fields of human-computer interactions. So there is a massive motivation to determine the "best" approach to do this tracking, and as such we sought to analyze two of the more recent and popular implementations of object tracking: CSRT and KCF.

The particular domain we worked in was with videos. To properly track an object, you must see both where it comes from and where it moves based off of time, so this format of analyzing videos was critical. In the problem space, one can think of videos as a set of frames aka images lined up based off of time. This allowed us to analyze rgb images from videos after the videos were separated into frames.

As stated, since we're using opencv’s library to implement these object tracking algorithms, the main difference in approach to this problem that our project tackles is tracking accuracy and efficiency of these algorithms as well as reading from a large dataset. Another difference in our approach is that the dataset was so large that we had to build our own metadata system to read directly from the zip file for each algorithm.


## Approach:
As we were looking to analyze object tracking algorithms, we decided that one of the most key factors would be breadth of data tested. The more data we tested, the more statistically accurate the results would be. We used opencvs implementation of CSRT (Channel and Spatial Reliability Tracking) which uses underlying histograms of oriented gradients and colors as well KCF (Kernelized Correlation Filters) which mainly relies on filtering and background subtraction. We quite uniquely linked the TrackingNet dataset and TrackingNet's metrics with our own metrics and opencv implementation by reading directly from the zipped set of videos (>2,500 videos totaling 92 GB) into our algorithm.

One problem was tracking speed without affecting the actual speed. TrackingNet holds no way of calculating the frames analyzed per second by the algorithm, so we carefully wrote scripts to track this, making sure not to impact the performance with the timing metric code . We also built a python generator that reduces the memory required to run this analysis, as it was maxing out our memory usage on our personal machines. The last major problem we solved was the massive 92 GB zipped file size for the >2,500 videos. We wrote a script that reads directly from the zip files to efficiently utilize memory (rather than unzipping the data to access). On our laptops, we barely have enough room for the zipped data, let alone the even larger unzipped data.

We made one judgement call. We acknowledged that the absolute wealth of data in TrackingNet must take its initial first frame object bounding box from its own algorithm rather than hand designed, so expecting 100% accuracy compared to computer-generated annotations would be impossible.

## Experiments and results:
We used a subset of the TrackingNet dataset (link in references) for our experimentation. The size of such was roughly >2,500 videos (92 GB zipped) and corresponding annotation files. For our use-case, the data was entirely test data and no training data.

Evaluation metrics we used included: Frames Tracked per Second, Success Average, Precision Average, and NPrecision Average. Taken directly from TrackingNet, the definition for the last three metrics are as follows (see TrackingNet metric paper in references): The success average is measured as the Intersection over Union (IoU) of the pixels between the ground truth bounding boxes and the ones generated by the tracker. The precision average is measured as the distance in pixels between the centers of the ground truth bounding box and the tracker's bounding box. The NPrecision Average is the precision average but normalized to take into account the size of the bounding box as different objects are different sizes in videos of course so just counting pixels won't be as accurate. Thus the more important of these two metrics are Success Average and NPrecision Average. But the most important and innovative out of all the metrics is our metric of how fast this object tracking can be accomplished for the algorithm.
All of these were in consistent conditions because they were run on the same 2018 Macbook Pro quad core machine with an Intel Core i7-8559U processor.


| Model Name | Success Average | Precision Average | NPrecision Average | Frames per second | Frames computed |
|------------|-----------------|-------------------|--------------------|-------------------|-----------------|
| CSRT       | 44.39142        | 40.23581          | 50.20777           | 22.49346          | 1179026         |
| KCF        | 34.99414        | 29.17041          | 37.6875            | 39.33432          | 1179026         |
{: .tablelines}


The only parameters necessary for this CSRT and KCF tracking were the video to track an object in and the bounding box for the object to be tracked in the first frame. There would not be enough time to individually adjust each bounding box, but for the sake of completeness we adjusted a few bounding boxes to notice effects of these parameters: in both CSRT and KCF, every metric went down in value when the bounding box was set to a random, unimportant part of the image, and there appeared to be no noticeable difference between the metrics on larger and smaller bounding boxes on of course equally larger or smaller objects. As for changing which video was input, video resolution and size seemed to have little to no impact on algorithm performance for CSRT while it was ever so slightly more important for KCF. 

Overall, we noticed a few trends in both algorithms’ results. For starters, occlusion took perfectly-working runs of the algorithm and threw them out the window. As the algorithm tracks the object from frame to frame, the object not existing in frame for a few frames destroys the accuracy. Additionally, rotation of the object proved a challenge for the algorithm as well. Texture alone was not enough for the algorithm to properly track the object. These results are to be expected to some degree. If there was a tracking algorithm designed that could easily handle the hurdles of object rotation as well as occlusion, then no other algorithm would be needed or ever used. 

Some individual trends that distinguished CSRT from KCF were: CSRT was way more resilient to similar objects entering frame that were not the object being tracked than KCF. Additionally, KCF was better at handling objects that disappeared from frames to reappear again later. While CSRT kept its old bounding box from where the object disappeared and waited for the object to return to that location, KCF removed the bounding box then readded it when the object was redetected.

In an abbreviated conclusion about the metrics, CSRT’s object tracking has a degree of accuracy and precision that KCF just cannot match while KCF is of a speed near double CSRT. There seems to be a clear tradeoff between efficiency and accuracy.


## Qualitative results:

#### CSRT:
Fully successful case:

![CSRT Example](img/sample3.gif){: .result}

Partially successful cases:

![CSRT Success Example](img/sample10.gif){: .result}
![CSRT Example](img/sample.gif){: .result}

Failure case:

![CSRT Example](img/sample6.gif){: .result}

*The blue boxes are the predicted bounding boxes, whereas the green is TrackingNet's ground truth*

#### KCF:
Fully successful case:

![KCF Success](img/kcf7.gif){: .result}

Partially successful case:

![KCF Example](img/kcf3.gif){: .result}

Failure cases:

![KCF Failure](img/kcf1.gif){: .result}
![KCF Failure](img/kcf2.gif){: .result}

*The blue boxes are the predicted bounding boxes, whereas the green is TrackingNet's ground truth*

#### Side by side comparisons:
![KCF Example](img/kcf5.gif){: .result} ![CSRT Example](img/csrt1.gif){: .result}
![KCF Example](img/kcf4.gif){: .result} ![CSRT Example](img/csrt4.gif){: .result}


## Conclusion and future work:
As already stated in the abstract, we came to appreciate CSRTs accuracy in object tracking. We were too dismissive of CSRT in the first update. This around 50% precision and accuracy proved its value side by side when compared to some of the more horrid trackings from KCF. However, we also appreciated how KCF took a fraction of the time to run in comparison to CSRT. The filters’ speed cannot be overstated. Overall, some common themes and findings we saw were:

Knowing the strengths and weaknesses of these CSRT and KCF object trackings could be very useful to companies or groups focused on self driving vehicles or surveillance because of the applications of computer vision in these fields. Of course, this is also generalizable to human-computer interaction as a whole, as having the most accurate tracking algorithm for any computer system involved in human affairs given that systems domain would be paramount. 

As for the biggest trade offs, from our results, we can more confidently trust CSRT as an accurate object tracking algorithm with notable exception cases such as rotation and occlusion. As such, we would highly recommend self driving vehicles and surveillance companies and other businesses where accuracy is paramount use this object tracking algorithm instead of KCF. Less than 50% success average might not be that much, but it is leagues better than 35% success average. Paradoxically, we would recommend that these surveillance and self driving vehicle companies use KCF to properly parse the massive amount of data their systems receive every second to efficiently analyze all the data. So in total, the major flaw of each object tracking algorithm is clear and complementary. So if we were to really step in front of Google building their self driving cars or in front of the pentagon for surveillance tracking, we would propose some sort of hybrid of the two to handle data both efficiently and accurately while of course noting that the accuracy and speed may not be perfect but they are excellent in comparison to alternatives.

Our future work on the subject would focus on selecting another recent and praised tracking algorithm with already implemented code, namely TLD (Tracking Learning Detection) and its variants. We would use TLD for comparing findings between both of the previously compared algorithms on the same dataset to better conclude how each algorithm would fit in use cases as well as the general appeal and features of each algorithm, an apples to apples comparison. 


## References:
[TrackingNet](https://tracking-net.org/)

[Paper On TrackingNet's metrics](https://openaccess.thecvf.com/content_ECCV_2018/papers/Matthias_Muller_TrackingNet_A_Large-Scale_ECCV_2018_paper.pdf)

[CSRT](https://arxiv.org/pdf/1611.08461)

[KCF](https://arxiv.org/abs/1404.7584)

<!-- [TLD](https://ieeexplore.ieee.org/document/6104061) and [e-TLD](https://arxiv.org/abs/2009.00855) -->
