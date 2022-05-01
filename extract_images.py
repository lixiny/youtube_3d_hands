import cv2
from pytube import YouTube

import argparse
import glob
import json
import os
from os.path import join

from video_lists import training_list, test_val_list

YT_ROOT = "https://youtu.be/"


def run(videos_id_list, out_dir_root="./data/youtube"):
    """Download images for each video in the list.

    Args:
        videos_id_list: The list of YouTube video IDs.
        out_dir_root: Output directory.
    """
    for video_id in videos_id_list:
        try:
            print(video_id)
            out_path = join(out_dir_root, video_id, "video")
            extract_frames(out_path)
            print(out_path, "Extract finished")
        except Exception as e:
            print(e)


def extract_frames(out_path):
    """Extract frames from the video.
       The method saves all frames, not just an annotated subset.

    Args:
        out_path: Filepath to the video directory.
    """
    vid_path = join(out_path, "raw.mp4")
    f_out_path = join(out_path, "frames")
    os.makedirs(f_out_path, exist_ok=True)


    vidcap = cv2.VideoCapture(vid_path, cv2.CAP_FFMPEG)
    count = 0
    vid_len = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_frames = len(glob.glob(join(out_path, "frames", "*.png")))
    print("video length: ", vid_len)
    
    if saved_frames >= vid_len: # sometimes the extracted frames have 1 frames more than the cv2.CAP_PROP_FRAME_COUNT
        print(f"{f_out_path} already has {saved_frames} frames, skip")
        return

    print(f"saved_frames {saved_frames}, vid_len {vid_len}")
    print("Extract frames.")

    try:
        os.system(f"ffmpeg -i {vid_path} -start_number 0 {f_out_path}/%d.png ")
    except Exception as e:
        print(e)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract YouTube video frames.")
    parser.add_argument("--vid", help="Video ID.", required=False, default=None, type=str)
    parser.add_argument("--set", help="Choose the whole set of IDs.", required=False, default=None, type=str)

    args = parser.parse_args()

    if args.vid is not None:
        run([args.vid])
    elif args.set == "train":
        run(training_list)
    elif args.set == "test":
        run(test_val_list)
