import os
import glob
import argparse
import zipfile

import tqdm
import numpy as np
import tracker
import cv2

from metrics import evaluate

def track(model:tracker.Tracker, folder:str, outdir:str):
    def write_bbox(bbox, f):
        f.write(','.join(map(str, bbox)) + '\n')

    anno_folder = os.path.join(folder, "anno/")
    image_folder = os.path.join(folder, "zips/")
    print("Folder: %s" % folder)
    for anno_file in tqdm.tqdm(glob.glob(os.path.join(anno_folder, "*.txt"))):
        base_name = os.path.basename(anno_file)[:-4]

        # get first bbox
        with open(anno_file, "r") as f:
            first_bbox = tuple(map(float,f.readline().split(",")))

        # get tracking
        with open(os.path.join(outdir, base_name + ".txt"), 'w') as f:
            # write first bbox
            write_bbox(first_bbox, f)

            # get corresponding .zip file
            z = zipfile.ZipFile(os.path.join(image_folder, base_name + ".zip"))

            names = [name for name in z.namelist() if name.endswith('.jpg')]
            names.sort(key=lambda s: int(s[:-4]))

            image_gen = (cv2.imdecode(np.frombuffer(z.read(name), np.uint8), 1) for name in names)
            for image, (_, bbox) in model.predict_frames(image_gen, bbox=first_bbox):
                if not _:
                    write_bbox((0,0,0,0), f) # failure case
                else:
                    write_bbox(bbox,f)

    print("\n=======\nAvg tracking time per frame: %s\n=======\n" % model.avg_time_per_frame())

    return model.avg_time_per_frame()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--folder", help="path to training folder with anno/ and zips/", type=str, required=True)
    parser.add_argument("-o", "--outdir", help="folder to write your subm.txt files", type=str, required=True)
    parser.add_argument("-s", "--summary", help="summary filepath", type=str, default="summary.csv")
    parser.add_argument("-m", "--model", help="model type ('CSRT', 'KCF', 'GOTURN')", type=str, choices=['CSRT', 'KCF', 'GOTURN'], required=True)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)


    model = tracker.factory(args.model, timed=True)

    track(model, args.folder, args.outdir)

    print("Evaluating...")
    Success_Average, Precision_Average, NPrecision_Average = evaluate(glob.glob(os.path.join(args.outdir, "*.txt")), glob.glob(os.path.join(args.folder, "anno", "*.txt")))

    if not os.path.exists(args.summary):
        with open(args.summary, 'w') as f: f.write('"Model","Folder name","Success Average","Precision Average","NPrecision_average","Frames per second","Frames computed"\n')

    with open(args.summary, 'a') as f:
        f.write(",".join(map(lambda s: '"%s"' % s, (args.model,
                                                    args.folder,
                                                    Success_Average,
                                                    Precision_Average,
                                                    NPrecision_Average,
                                                    1/model.avg_time_per_frame(),
                                                    model.frames))) + "\n")