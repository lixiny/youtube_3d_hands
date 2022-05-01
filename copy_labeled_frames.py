import json
import time
import cv2
import os
import shutil
from tqdm import tqdm


def load_dataset(fp_data='./data/youtube_val.json'):
    """Load the YouTube dataset.

    Args:
        fp_data: Filepath to the json file.

    Returns:
        Hand mesh dataset.
    """
    with open(fp_data, "r") as file:
        data = json.load(file)

    return data


def copy_a_sample(data, ann_index):
    """Retrieve an annotation-image pair from the dataset.

    Args:
        data: Hand mesh dataset.
        ann_index: Annotation index.

    Returns:
        A sample from the hand mesh dataset.
    """
    ann = data['annotations'][ann_index]
    images = data['images']
    img_idxs = [im['id'] for im in images]

    img = images[img_idxs.index(ann['image_id'])]
    img_path = os.path.join("data", img['name'])

    new_img_path = img_path.replace('youtube', "youtube_annotated")
    if os.path.exists(new_img_path):
        return
    new_img_dir = os.path.dirname(new_img_path)
    os.makedirs(new_img_dir, exist_ok=True)
    shutil.copyfile(img_path, new_img_path)


def copy_split_samples(data_sp):
    print("Data keys:", [k for k in data_sp.keys()])
    print("Image keys:", [k for k in data_sp['images'][0].keys()])
    print("Annotations keys:", [k for k in data_sp['annotations'][0].keys()])
    print("The number of images:", len(data_sp['images']))
    print("The number of annotations:", len(data_sp['annotations']))
    print(f"The total len of data subset: {len(data_sp['annotations'])}")
    for i in tqdm(range(len(data_sp['annotations']))):
        copy_a_sample(data_sp, i)


if __name__ == "__main__":
    NEW_ROOT = "data/youtube_annotated"
    os.makedirs(NEW_ROOT, exist_ok=True)

    splits = ["val", "test", "train"]
    for sp in splits:
        print(f">>>>>>>>>>>>>>>>>> {sp} >>>>>>>>>>>>>>>>>>>")
        fp_data = os.path.join("data", f"youtube_{sp}.json")
        data = load_dataset(fp_data)
        copy_split_samples(data)
