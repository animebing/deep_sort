# vim: expandtab:ts=4:sw=4
from __future__ import print_function, absolute_import
import os
import argparse
import numpy as np
import cv2

from application_util import visualization
from deep_sort.detection import Detection


def gather_sequence_info(sequence_dir):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}

    detection_file = os.path.join(sequence_dir, "det/det.txt")
    detections = np.loadtxt(detection_file, delimiter=',')

    score_max = np.max(detections[:, 6])
    score_min = np.min(detections[:, 6])
    #print("max score: %f, min score: %f" % (score_max, score_min))


    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections with ten columns
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[detection.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence = row[2:6], row[6]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, [1.0]))
    return detection_list


def draw_detections(seq_dir, threshold):
    """ Draw detection results a sequence with threshold for detection

    Parameters
    ----------
    seq_dir: str
        the sequence directory for visualization
    threshold: float
        score above the threshold will be shown

    """
    seq_info = gather_sequence_info(seq_dir)


    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, 0.0)

        detections = [d for d in detections if d.confidence >= threshold]



        image = cv2.imread(
            seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        vis.set_image(image.copy())
        vis.draw_detections(detections)

    visualizer = visualization.Visualization(seq_info, update_ms=seq_info['update_ms'])
    visualizer.run(frame_callback)

def parse_args():
    """ Parse command line arguments

    """
    parser = argparse.ArgumentParser(description="visualization of different detection methods with threshold")
    parser.add_argument('-s', "--seq_dir", type=str, required=True,
                        help="the sequence directory to be visualized")
    parser.add_argument('-t', "--threshold", type=float, required=True,
                        help="the threshold for detection result for visualization")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    draw_detections(args.seq_dir, args.threshold)

if __name__ == "__main__":
    main()

    """
    python draw_detections.py -s MOT17/train/MOT17-02-DPM -t -1.0
    """