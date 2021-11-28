import json
import os
import random
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms.functional as FT
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


# start from 1, not 0
label_map = {k: v+1 for v, k in enumerate(voc_labels)}

# Color map for bounding box
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']

# label: distinct_color
label_color_map = dict(zip(voc_labels, distinct_colors))


# dev root path: /mnt/e/DataHub/voc/VOCdevkit/VOC2012/Annotations/
# example annotation is 6 bbox, 6 labels, 6 difficulties
def parse_annotation(annotation_path='/mnt/e/DataHub/voc/VOCdevkit/VOC2012/Annotations/2007_000129.xml'):
    """
    input: /mnt/e/DataHub/voc/VOCdevkit/VOC2012/Annotations/2007_000129.xml

    output:
    {'boxes': [[69, 201, 254, 499], [250, 241, 333, 499], [0, 143, 66, 435], [0, 0, 65, 362], [73, 0, 271, 461], [251, 18, 333, 486]],
     'labels': [2, 2, 2, 15, 15, 15],
     'difficulties': [0, 1, 1, 1, 0, 0]}
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    labels = []
    difficulties = []

    for object in root.iter('object'):
        difficult = int(object.find('difficult').text == '1') # true / false -> int
        label = object.find('name').text.lower().strip()      # name as lower char
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)
    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}



def create_data_lists(voc07_path, voc12_path, output_folder):
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = []
    train_objects = []
    n_objects = 0

    for path in [voc07_path, voc12_path]:
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()
            # Example
            # 2008_000002
            # 2008_000003
            # 2008_000007
            # 2008_000008

        for id in tqdm(ids):
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml')) # objects: dict
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images) # res lists comparison

    # save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)

    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)

    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j) # global variable

    print(f'There are {len(train_images)} training images containig a total of {n_objects} objects. Files have been saved to {os.path.abspath(output_folder)}')

    # Test data
    test_images = []
    test_objects = []
    n_objects = 0

    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()

    for id in tqdm(ids):
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects) == 0:
            continue

        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))

    assert len(test_images) == len(test_objects)

    print('\nThere are %d test images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))


def photometric_distort(image):
    new_image = image
    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]
    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                adjust_factor = random.uniform(0.5, 1.5)
        new_image = d(new_image, adjust_factor)
    return new_image


def expand(image, boxes, filler):
    """
    perform a zooming out operation by placing the image in larger canvas of filler meterial

    image: (3, origianal_h, original_w)
    boxes: (n_objects, 4)
        boxes: [xmin, ymin, xmax, ymax]
    filler: [R, G, B]
    return : expanded image, updated bounding box coordinates

    """
    original_h = image.size(1) # 224
    original_w = image.size(2) # 224
    max_scale = 4
    scale = random.uniform(1, max_scale) # 2

    new_h = int(scale * original_h)  # 448
    new_w = int(scale * original_w)  # 448

    filler = torch.FloatTensor(filler) # (3)
    # torch.unsqueeze() ~ tf.expand_dims()
    #               shape: 3                                        [3,1]                 [3, 1, 1]
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # [3, 448, 448] value as 1

    # place original image at random coordinate in new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)  # randint(0, 448-224) ~ randint(0, 224) ~ e.g) 110
    right = left + original_w  # 110 + 224

    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image # inserting image to new canvas

    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0) # xmin, ymin, xmax, ymax + left, top, left, top

    return new_image, new_boxes


def find_intersection(set_1, set_2):
    # set: [xmin, ymin, xmax, ymax]
    # xmax, ymax
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0)) # (n1, n2, 2)
    # Annatomy: [b, 2] (xmin, ymin) -> [b, 1, 2] ||  [b, 2] (xmin, ymin) -> [1, 1, 2]

    # xmin, ymin
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0)) # (n1, n2, 2)

    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)

    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def find_jaccard_overlap(set_1, set_2):
    intersection = find_intersection(set_1, set_2)

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1]) # (n1)

    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1]) # (n2)

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection

    return intersection / union

def random_crop(image, boxes, labels, difficulties):
    original_h = image.size(1)
    original_w = image.size(2)

    while True:
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None]) # none is not perform cropping

        if min_overlap is None:
            return image, boxes, labels, difficulties

        max_trials = 50
        for _ in range(max_trials):
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # aspect ratio has to be in [0.5, 2]
            # TERMINOLOGY: Aspect Ratio = image is the ratio of its width to its height ~ 16:9
            # x = width, y = height (standard, width is more bigger)
            # r = y / x
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # crop coordinate
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom]) # (4)

            overlap = find_jaccrard_overlap(crop.unsqueeze(0), boxes)
            overlap = overlap.squeeze(0)

            if overlap.max().item() < min_overlap:
                continue

            new_image = image[:, top:bottom, left:right]
            bb_center = (boxes[:, :2] + boxes[:, 2:]) / 2.

            centers_in_crop = (bb_center[:, 0] > left) * (bb_center[:, 0] < right) * (bb_center[:, 1] > top) * (bb_center[:, 1] < bottom)

            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    new_image = FT.hflip(image)
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    return new_iamge, new_boxes



def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    new_image = FT.resize(image, dims)

    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes + new_dims
    return new_image, new_boxes



# main transform
# the data requires to transform both [image and bbox]
def transform(image, boxes, labels, difficuties, split):
    assert split in {'TRAIN', 'TEST'}
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficuties

    if split == 'TRAIN':
        new_image = photometric_distort(new_image)

        new_image = FT.to_tensor(new_image)

        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels, new_difficulties)

        new_image = FT.to_pil_image(new_image)

        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

        new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

        new_image = FT.to_tensor(new_image)
        new_image = FT.normalize(new_image, mean=mean, std=std) # mean and std from global

        return new_iamge, new_boxes, new_difficulties
