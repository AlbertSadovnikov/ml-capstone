import os
import cv2
import numpy as np


download_path = './download'
annotated_suffix = '_Annotated_Cars.png'
negative_suffix = '_Annotated_Negatives.png'

storage_folder = './images'
image_file_format = '%06d.png'
annotation_file = 'dataset.xml'

os.makedirs(storage_folder, exist_ok=True)

downloaded_files = os.listdir(download_path)

annotated_cars = [item[:-len(annotated_suffix)] for item in downloaded_files if item.endswith(annotated_suffix)]
annotated_negatives = [item[:-len(negative_suffix)] for item in downloaded_files if item.endswith(negative_suffix)]

# get only ones which have annotations
items_all = sorted(set(annotated_cars + annotated_negatives))

# tile images, so they are small enough for training
frame = (512, 512)


def tile_1d(length, chunk_size):
    n_tiles = (length - 1) // chunk_size + 1
    starts = np.int32(np.round(np.linspace(0, length - chunk_size, n_tiles)))
    ends = [length] if length <= chunk_size else starts + chunk_size
    for item in zip(starts, ends):
        yield item


def tile_2d(size, chunk_size):
    for tile_0 in tile_1d(size[0], chunk_size[0]):
        for tile_1 in tile_1d(size[1], chunk_size[1]):
            yield tile_0 + tile_1

image_count = 0

with open(annotation_file, 'w') as fp:
    fp.write("""<?xml version='1.0' encoding='ISO-8859-1'?>
<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>
<dataset>
<name>imglab dataset</name>
<comment>Created by imglab tool.</comment>
<images>""")

    for item in items_all:
        image_path = os.path.join(download_path, item + '.png')
        if not os.path.exists(image_path):
            print('Base image for %s does not exist', item)
            continue
        print('Processing %s' % item)
        w, h = 40, 40
        large_image = cv2.imread(image_path)

        car_path = os.path.join(download_path, item + annotated_suffix)
        negative_path = os.path.join(download_path, item + negative_suffix)

        has_cars = os.path.exists(car_path)
        has_negatives = os.path.exists(negative_path)

        if not has_cars and not has_negatives:
            print('annotations for %s do not exist', item)
            continue

        car_image = np.zeros(large_image.shape[:2], dtype=np.uint8)
        if has_cars:
            car_image = np.amax(cv2.imread(car_path), axis=2)

        negative_image = np.zeros(large_image.shape[:2], dtype=np.uint8)
        if has_negatives:
            negative_image = np.amax(cv2.imread(negative_path), axis=2)
        fp.write("<!-- tiles from %s -->\n" % item)
        for tile in tile_2d(large_image.shape[:2], frame):
            img_tile = large_image[tile[0]:tile[1], tile[2]:tile[3], :3]
            car_y, car_x = car_image[tile[0]:tile[1], tile[2]:tile[3]].nonzero()
            neg_y, neg_x = negative_image[tile[0]:tile[1], tile[2]:tile[3]].nonzero()
            if len(car_x) == 0 and len(neg_x) == 0:
                continue
            image_filename = os.path.join(storage_folder, image_file_format % image_count)
            cv2.imwrite(image_filename, img_tile)
            fp.write("<image file='%s'>\n" % image_filename)
            fp.write("<!-- tile %d %d %d %d -->\n" % (tile[2], tile[3], tile[0], tile[1]))
            for x, y in zip(car_x, car_y):
                fp.write("<box top='%d' left='%d' width='%d' height='%d'><label>car</label></box>\n"
                         % (y - h / 2, x - w / 2, w, h))

            for x, y in zip(neg_x, neg_y):
                fp.write("<box top='%d' left='%d' width='%d' height='%d'><label>background</label></box>\n"
                         % (y - h / 2, x - w / 2, w, h))
            fp.write("</image>\n")
            image_count += 1
    fp.write("""</images>
</dataset>""")
    fp.close()
