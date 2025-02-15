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
            out_path = download_video(video_id, out_dir_root)
            print(out_path, "Download finished")
        except Exception as e:
            print(e)


def download_video(yt_id, out_dir_root):
    """Download a video.

    Args:
        yt_id: YouTube video ID.
        out_dir_root: Output directory.

    Returns:
        Filepath to the video directory.
    """
    out_path = join(out_dir_root, yt_id, "video")
    os.makedirs(out_path, exist_ok=True)

    vid_path = join(out_path, "raw.mp4")

    if not os.path.exists(vid_path):
        print("Download a video.")
        YouTube(join(YT_ROOT, yt_id)).streams.filter(subtype="mp4", only_video=True).order_by(
            "resolution"
        ).desc().first().download(out_path)

        os.rename(join(out_path, os.listdir(out_path)[0]), vid_path)

    return out_path



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
