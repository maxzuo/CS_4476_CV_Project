import os
import glob
import argparse
import zipfile
import platform
import multiprocessing
from multiprocessing import Pool

import tqdm
import numpy as np
import tracker
import cv2

from metrics import evaluate

def track_video(model:str, anno_file:str, folder:str, outdir:str) -> tracker.Tracker:
    def write_bbox(bbox, f):
        f.write(','.join(map(str, bbox)) + '\n')
    model = tracker.factory(model, timed=True)

    anno_folder = os.path.join(folder, "anno/")
    image_folder = os.path.join(folder, "zips/")
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

    return model.time, model.ptime, model.frames

def unpack_track(args):
    return track_video(*args)

def track(model:str, folder:str, outdir:str, processes:int=1):

    anno_folder = os.path.join(folder, "anno/")
    image_folder = os.path.join(folder, "zips/")
    print("Folder: %s" % folder)

    anno_files = glob.glob(os.path.join(anno_folder, "*.txt"))
    time = 0.
    ptime = 0.
    frames = 0
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')
    with Pool(processes=processes) as p:
        with tqdm.tqdm(total=len(anno_files)) as pbar:
            args = [(model, anno_files[i], folder, outdir) for i in range(len(anno_files))]
            for avg_time, avg_cpu, _frames in p.imap_unordered(unpack_track, args):
                time += avg_time
                ptime += avg_cpu
                frames += _frames
                pbar.update()

    print("\n=======\nAvg tracking time per frame: %s\n=======\n" % (time / frames))


    return time / ptime, ptime / frames, frames


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--folder", help="path to training folder with anno/ and zips/", type=str, required=True)
    parser.add_argument("-o", "--outdir", help="folder to write your subm.txt files", type=str, required=True)
    parser.add_argument("-s", "--summary", help="summary filepath", type=str, default="summary.csv")
    parser.add_argument("-m", "--model", help="model type ('CSRT', 'KCF', 'GOTURN')", type=str, choices=['CSRT', 'KCF', 'GOTURN'], required=True)
    parser.add_argument("-p", "--processors", help=f"number of processors to be used. You have: {os.cpu_count()} logical processors", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Starting track...")
    avg_time, avg_cpu_time, frames = track(args.model, args.folder, args.outdir, processes=args.processors)

    print("Evaluating...")
    Success_Average, Precision_Average, NPrecision_Average = evaluate(sorted(glob.glob(os.path.join(args.outdir, "*.txt"))), sorted(glob.glob(os.path.join(args.folder, "anno", "*.txt"))))

    if not os.path.exists(args.summary):
        with open(args.summary, 'w') as f: f.write('"Model","Folder name","Success Average","Precision Average","NPrecision_average","Frames per second","CPU Usage per frame", "Frames computed"\n')

    with open(args.summary, 'a') as f:
        f.write(",".join(map(lambda s: '"%s"' % s, (args.model,
                                                    args.folder,
                                                    Success_Average,
                                                    Precision_Average,
                                                    NPrecision_Average,
                                                    1/avg_time,
                                                    avg_cpu_time,
                                                    frames))) + "\n")
