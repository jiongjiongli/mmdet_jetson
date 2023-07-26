import json
from pathlib import Path
import shutil

from shapely.geometry import Polygon
from PIL import Image


class CocoFormat:
    @staticmethod
    def create_category_info(category_id, category_name):
        category_info = {'id': category_id,
                         'name': category_name}
        return category_info

    @staticmethod
    def create_image_info(image_id,
                          image_file_name,
                          image_height,
                          image_width):
        image_info = {'id': image_id,
                      'file_name': image_file_name,
                      'height': image_height,
                      'width': image_width}

        return image_info

    @staticmethod
    def create_annotation_info(annotation_id,
                               image_id,
                               category_id,
                               is_crowd,
                               bbox,
                               segmentation,
                               area):
        annotation_info = {'id': annotation_id,
                           'image_id': image_id,
                           'category_id': category_id,
                           'iscrowd': is_crowd,
                           'bbox': bbox,
                           'area': area,
                           'segmentation': segmentation}

        return annotation_info

def convert(image_dir_path,
            via_ann_file_path,
            category_names,
            output_images_dir_path=None,
            output_ann_file_path=None,
            first_category_index=1):
    category_dict = {}

    coco_categories = []

    for category_id, category_name in enumerate(category_names, start=first_category_index):
        category_dict[category_name] = category_id
        coco_category = CocoFormat.create_category_info(category_id, category_name)
        coco_categories.append(coco_category)

    with open(Path(via_ann_file_path).as_posix(), 'r') as file_stream:
        via_ann_data = json.load(file_stream)

    coco_images = []
    coco_annotations = []

    image_dir_path = Path(image_dir_path)
    default_category_name = category_names[0]
    image_id = 0
    annotation_id = 0

    for via_ann_key, via_ann in via_ann_data.items():
        image_file_name = via_ann['filename']
        image_file_path = image_dir_path / image_file_name

        assert image_file_path.exists(), r'Error! image_file_path {} not exist!'.format(image_file_path)

        if output_images_dir_path is not None:
            output_images_dir_path.mkdir(parents=True, exist_ok=True)

            output_image_file_path = output_images_dir_path / image_file_path.name
            shutil.copyfile(image_file_path.as_posix(), output_image_file_path.as_posix())

        with Image.open(image_file_path.as_posix()) as image:
            image_width, image_height = image.size

        image_info = CocoFormat.create_image_info(
                              image_id,
                              Path(image_file_name).name,
                              image_height,
                              image_width
        )

        coco_images.append(image_info)

        via_regions = via_ann['regions']

        for via_region_index, via_region in via_regions.items():
            region_attributes = via_region['region_attributes']
            category_name = region_attributes.get('label', default_category_name)
            category_id = category_dict.get(category_name)

            if category_id is None:
                warn_info_format = r'Warning: Ignore because category_name {} from image_file_name {} not in category_names {}'
                warn_info = warn_info_format.format(category_name,
                                  image_file_name,
                                  category_names)
                print(warn_info)
                continue

            is_crowd = 0

            shape_attributes = via_region['shape_attributes']
            all_points_x = shape_attributes['all_points_x']
            all_points_y = shape_attributes['all_points_y']

            points = [(x, y) for x, y in zip(all_points_x, all_points_y)]
            polygon = Polygon(points)

            x_min, y_min, x_max, y_max = polygon.bounds
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            segmentation = []

            for x, y in points:
                segmentation.extend([x, y])

            area = polygon.area

            coco_annotation = CocoFormat.create_annotation_info(annotation_id,
                                       image_id,
                                       category_id,
                                       is_crowd,
                                       bbox,
                                       segmentation,
                                       area)
            coco_annotations.append(coco_annotation)
            annotation_id += 1

        image_id += 1


    coco_output = {'images': coco_images,
                   'categories': coco_categories,
                   'annotations': coco_annotations}

    if output_ann_file_path is not None:
        output_ann_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(Path(output_ann_file_path).as_posix(), 'w') as file_stream:
            json.dump(coco_output, file_stream, indent=4)

    return coco_output


def main():
    category_names = ['balloon']
    data_path_infos = [{
                        'image_dir_path': r'data/balloon/train',
                        'output_ann_file_name': 'trainval.json'
                       },
                       {
                        'image_dir_path': r'data/balloon/val',
                        'output_ann_file_name': 'test.json'
                       }]

    outout_dir_path = Path('data/balloon_dataset')
    output_images_dir_path = outout_dir_path / 'images'
    output_ann_dir_path = outout_dir_path / 'annotations'

    for data_path_info in data_path_infos:
        image_dir_path = Path(data_path_info['image_dir_path'])
        via_ann_file_path = image_dir_path / 'via_region_data.json'

        output_ann_file_name = data_path_info['output_ann_file_name']
        output_ann_file_path = output_ann_dir_path / output_ann_file_name

        convert(image_dir_path,
                via_ann_file_path,
                category_names,
                output_images_dir_path=output_images_dir_path,
                output_ann_file_path=output_ann_file_path)


if __name__ == '__main__':
    main()
