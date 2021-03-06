import tracker
import cv2
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", type=str, help="filepath to video you would like to extract (0 for webcam)", required=True)

    args = parser.parse_args()

    # open video
    if args.filepath == '0':
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(args.filepath)

    # make iterator object for video (not necessary if list of frames)
    video = tracker.VideoIterator(video)
    first_bbox = (50, 50, 350, 350) # random bbox

    # instantiate tracker
    track = tracker.CSRT(timed=True) #tracker.KCF() # tracker.GOTURN()

    # process video
    index = 1
    for image, (_, bbox) in track.predict_frames(video, bbox=first_bbox):
        if not index % 100:
            print(track.avg_time_per_frame()) # print average time per frame every 100 frames
        index += 1
        # for drawing bounding box
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        print(bbox)
        cv2.rectangle(image, p1, p2, (255,0,0), 2, 1)

        cv2.imshow("test", image)
        break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

