import os
import glob
import argparse
import zipfile
import imageio

import tqdm
import numpy as np
import tracker
import cv2

def create_gifs(model, anno_path, zip_path, gif_path, dim=(408,360)):

    z = zipfile.ZipFile(os.path.join(zip_path))

    names = [name for name in z.namelist() if name.endswith('.jpg')]
    names.sort(key=lambda s: int(s[:-4]))

    image_gen = (cv2.imdecode(np.frombuffer(z.read(name), np.uint8), 1) for name in names)

    success_images = []
    original_images = []
    precision_images = []


    with open(anno_path, "r") as f:
        first_bbox = tuple(map(float,f.readline().split(",")))
        for image, (_, bbox) in tqdm.tqdm(model.predict_frames(image_gen, bbox=first_bbox)):
            anno_bbox = tuple(map(float,f.readline().split(",")))

            p1 = (int(anno_bbox[0]), int(anno_bbox[1]))
            p2 = (int(anno_bbox[0] + anno_bbox[2]), int(anno_bbox[1] + anno_bbox[3]))

            m1 = (int(bbox[0]), int(bbox[1]))
            m2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

            c1 = (int(anno_bbox[0] + anno_bbox[2] // 2), int(anno_bbox[1] + anno_bbox[3] // 2))
            c2 = (int(bbox[0] + bbox[2] // 2), int(bbox[1] + bbox[3] // 2))

            original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_images.append(original)
            copy = original.copy()

            cv2.rectangle(copy, p1, p2, (0,255,0), 2, 1)
            cv2.rectangle(copy, m1, m2, (0,0,255), 2, 1)

            # cover success image
            success_image = np.zeros(copy.shape, np.uint8)

            minx = max(p1[0], m1[0])
            miny = max(p1[1], m1[1])

            maxx = min(m2[0], p2[0])
            maxy = min(m2[1], p2[1])

            cv2.rectangle(success_image, p1, p2, (200,200,0), -1)
            cv2.rectangle(success_image, m1, m2, (200,200,0), -1)
            cv2.rectangle(success_image, (minx, miny), (maxx, maxy), (0,200,200), -1)

            cv2.rectangle(success_image, p1, p2, (0,255,0), 2, 1)
            cv2.rectangle(success_image, m1, m2, (0,0,255), 2, 1)

            alpha = 0.6
            overlay = cv2.addWeighted(success_image, alpha, copy, 1-alpha, gamma=0)

            success_images.append(success_image)

            # precision
            precision_image = np.zeros(original.shape, dtype=np.uint8)
            prec_copy = original.copy()

            cv2.rectangle(precision_image, p1, p2, (0,255,0), 1)
            cv2.rectangle(precision_image, m1, m2, (0,0,255), 1)

            cv2.line(precision_image, c1, c2, (255,0,0), 5,1)

            cv2.rectangle(original, p1, p2, (0,255,0), 2, 1)
            cv2.rectangle(original, m1, m2, (0,0,255), 2, 1)

            overlay = cv2.addWeighted(precision_image, alpha, prec_copy, 1-alpha, gamma=0)
            precision_images.append(precision_image)


    imageio.mimsave(gif_path+"_success.gif", success_images)
    imageio.mimsave(gif_path+"_precision.gif", precision_images)
    imageio.mimsave(gif_path+"_original.gif", original_images)

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
    parser.add_argument("-g", "--gif", action='store_true')

    args = parser.parse_args()

    model = tracker.factory(args.model, timed=True)

    if not args.gif:
        create_video(model, args.annopath, args.zippath, args.vidpath, model_type=args.model)
    else:
        create_gifs(model, args.annopath, args.zippath, args.vidpath)