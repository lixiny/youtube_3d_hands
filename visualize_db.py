import json
import time
import cv2


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


def retrieve_sample(data, ann_index):
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
    return ann, img


def viz_sample(data, ann_index, faces=None, db_root='./data/'):
    """Visualize a sample from the dataset.

    Args:
        data: Hand mesh dataset.
        ann_index: Annotation index.
        faces: MANO faces.
        db_root: Filepath to the youtube parent directory.
    """
    import imageio
    import matplotlib.pyplot as plt
    import numpy as np
    from os.path import join

    ann, img = retrieve_sample(data, ann_index)

    image_name = img['name'].replace("youtube", "youtube_annotated")
    image = imageio.imread(join(db_root, image_name))
    vertices = np.array(ann['vertices'])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for j in range(vertices.shape[0]):
        v = vertices[j].astype(np.int32)
        cv2.circle(image, (v[0], v[1]), radius=2, thickness=-1, color=(0, 0, 255))

    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    return image


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract YouTube video frames.")
    parser.add_argument("--set",
                        help="Choose the whole set of IDs.",
                        required=False,
                        default="val",
                        choices=["train", "val", "test"],
                        type=str)

    args = parser.parse_args()
    data = load_dataset(fp_data=f"data/youtube_{args.set}.json")

    print("Data keys:", [k for k in data.keys()])
    print("Image keys:", [k for k in data['images'][0].keys()])
    print("Annotations keys:", [k for k in data['annotations'][0].keys()])

    print("The number of images:", len(data['images']))
    print("The number of annotations:", len(data['annotations']))

    print(f"The total len of data subset: {len(data['annotations'])}")
    for i in range(len(data['annotations'])):
        image = viz_sample(data, i)
        cv2.imshow('image', image)
        cv2.waitKey(0)  # interactive visualize
