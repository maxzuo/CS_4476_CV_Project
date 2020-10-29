import os
import glob
import argparse
import zipfile

import tqdm
import numpy as np
import tracker
import cv2

def create_video(model, anno_path, zip_path, vid_path, dim=(480,360), model_type="CSRT"):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(vid_path, fourcc, 20, dim)

    z = zipfile.ZipFile(os.path.join(zip_path))

    names = [name for name in z.namelist() if name.endswith('.jpg')]
    names.sort(key=lambda s: int(s[:-4]))

    image_gen = (cv2.imdecode(np.frombuffer(z.read(name), np.uint8), 1) for name in names)

    with open(anno_path, "r") as f:
        first_bbox = tuple(map(float,f.readline().split(",")))
        for image, (_, bbox) in tqdm.tqdm(model.predict_frames(image_gen, bbox=first_bbox)):
            anno_bbox = tuple(map(float,f.readline().split(",")))

            p1 = (int(anno_bbox[0]), int(anno_bbox[1]))
            p2 = (int(anno_bbox[0] + anno_bbox[2]), int(anno_bbox[1] + anno_bbox[3]))

            cv2.rectangle(image, p1, p2, (0,255,0), 2, 1)
            cv2.putText(image, 'Grnd Truth', (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

            cv2.rectangle(image, p1, p2, (255,0,0), 2, 1)
            cv2.putText(image, model_type, (p2[0] - 40, p2[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

            out.write(image)

    out.release()
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--zippath", help="zipfile of images", type=str, required=True)
    parser.add_argument("-a", "--annopath", help="path to annotation file", type=str, required=True)
    parser.add_argument("-v", "--vidpath", help="video out filepath", type=str, default="summary.csv")
    parser.add_argument("-m", "--model", help="model type ('CSRT', 'KCF', 'GOTURN')", type=str, choices=['CSRT', 'KCF', 'GOTURN'], required=True)

    args = parser.parse_args()

    model = tracker.factory(args.model, timed=True)

    create_video(model, args.annopath, args.zippath, args.vidpath, model_type=args.model)