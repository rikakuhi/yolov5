import os
import random
import xml.etree.ElementTree as ET
import numpy as np

classes_path = './my_data_classes'
train_percent = 1
image_path = r'E:\code\deep_learning\two_stage\faster_rcnn\image\Annotations700'

my_data_sets = ['train', 'val']


def get_classes(classes_path):
    """获取种类"""
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


classes, _ = get_classes(classes_path)

photo_nums = np.zeros(len(my_data_sets))
nums = np.zeros(len(classes))  # 用来统计每一种 类别有多少个object


def convert_annotation(image_id, list_file):
    in_file = open(os.path.join(image_path, '%s.xml' % image_id), encoding='utf-8')

    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        sizes = root.iter("size")
        for size in sizes:
            img_width = float(size.find("width").text)
            img_height = float(size.find("height").text)
        cls_id = str(classes.index(cls))
        xmlbox = obj.find('bndbox')
        x_min = max(int(float(xmlbox.find('xmin').text)), 0)
        y_min = max(int(float(xmlbox.find('ymin').text)), 0)
        x_max = min(int(float(xmlbox.find('xmax').text)), 1280)
        y_max = min(int(float(xmlbox.find('ymax').text)), 720)
        width = str((x_max - x_min) / img_width)
        height = str((y_max - y_min) / img_height)
        x_center = str((x_min + x_max) / 2 / img_width)
        y_center = str((y_max + y_min) / 2 / img_height)

        list_file.write(cls_id + " " + x_center + " " + y_center + " " + width + " " + height)
        list_file.write("\n")
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


def get_img_ids(percent, path=r"E:\code\deep_learning\two_stage\faster_rcnn\image\images700"):
    img_list = os.listdir(path)
    img_list = [_id[:-4] for _id in img_list]
    train_ids = []
    val_ids = []
    for i in range(int(len(img_list) * percent)):
        train_ids.append(img_list.pop(img_list.index(random.choice(img_list))))
    for val_id in img_list:
        val_ids.append(val_id)
    return train_ids, val_ids


if __name__ == "__main__":
    random.seed(0)
    ids = get_img_ids(train_percent)
    print("Generate train.txt and val.txt for train.")
    type_index = 0
    for image_ids, image_set in zip(ids, my_data_sets):

        # list_file = open('%s.txt' % image_set, 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file = open('./datasets/labels/%s.txt' % image_id, 'w', encoding='utf-8')
            # list_file.write("./image/images700/" + image_id + ".jpg")

            convert_annotation(image_id, list_file)
            list_file.write('\n')
            list_file.close()

        photo_nums[type_index] = len(image_ids)
        type_index += 1
    print("Generate 2007_train.txt and 2007_val.txt for train done.")
