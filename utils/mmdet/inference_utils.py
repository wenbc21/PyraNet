from typing import Literal, Union

def process_mmdet_results(mmdet_results: list,
                          cat_id: int = 0,
                          multi_person: bool = True) -> list:
    """Process mmdet results, sort bboxes by area in descending order.

    Args:
        mmdet_results (list):
            Result of mmdet.apis.inference_detector
            when the input is a batch.
            Shape of the nested lists is
            (n_frame, n_category, n_human, 5).
        cat_id (int, optional):
            Category ID. This function will only select
            the selected category, and drop the others.
            Defaults to 0, ID of human category.
        multi_person (bool, optional):
            Whether to allow multi-person detection, which is
            slower than single-person. If false, the function
            only assure that the first person of each frame
            has the biggest bbox.
            Defaults to True.

    Returns:
        list:
            A list of detected bounding boxes.
            Shape of the nested lists is
            (n_frame, n_human, 5)
            and each bbox is (x, y, x, y, score).
    """
    ret_list = []
    only_max_arg = not multi_person
    # for _, frame_results in enumerate(mmdet_results):
    cat_bboxes = mmdet_results[cat_id]
    # import pdb; pdb.set_trace()
    sorted_bbox = qsort_bbox_list(cat_bboxes, only_max_arg)

    if only_max_arg:
        ret_list.append(sorted_bbox[0:1])
    else:
        ret_list.append(sorted_bbox)
    return ret_list


def process_mmdet_results_TTSH(img_height, 
                          img_width,
                          mmdet_results: list,
                          cat_id: int = 0,
                          multi_person: bool = True, 
                          ) -> list:
    """Process mmdet results, sort bboxes by area in descending order.

    Args:
        mmdet_results (list):
            Result of mmdet.apis.inference_detector
            when the input is a batch.
            Shape of the nested lists is
            (n_frame, n_category, n_human, 5).
        cat_id (int, optional):
            Category ID. This function will only select
            the selected category, and drop the others.
            Defaults to 0, ID of human category.
        multi_person (bool, optional):
            Whether to allow multi-person detection, which is
            slower than single-person. If false, the function
            only assure that the first person of each frame
            has the biggest bbox.
            Defaults to True.

    Returns:
        list:
            A list of detected bounding boxes.
            Shape of the nested lists is
            (n_frame, n_human, 5)
            and each bbox is (x, y, x, y, score).
    """
    ret_list = []
    only_max_arg = not multi_person
    # for _, frame_results in enumerate(mmdet_results):
    cat_bboxes = mmdet_results[cat_id]
    # import pdb; pdb.set_trace()
    sorted_bbox = sort_bbox_list_TTSH(img_height, img_width, cat_bboxes, only_max_arg)

    if only_max_arg:
        ret_list.append(sorted_bbox[0:1])
    else:
        ret_list.append(sorted_bbox)
    return ret_list


def qsort_bbox_list(bbox_list: list,
                    only_max: bool = False,
                    bbox_convention: Literal['xyxy', 'xywh'] = 'xyxy'):
    """Sort a list of bboxes, by their area in pixel(W*H).

    Args:
        input_list (list):
            A list of bboxes. Each item is a list of (x1, y1, x2, y2)
        only_max (bool, optional):
            If True, only assure the max element at first place,
            others may not be well sorted.
            If False, return a well sorted descending list.
            Defaults to False.
        bbox_convention (str, optional):
            Bbox type, xyxy or xywh. Defaults to 'xyxy'.

    Returns:
        list:
            A sorted(maybe not so well) descending list.
    """
    # import pdb; pdb.set_trace()
    if len(bbox_list) <= 1:
        return bbox_list
    else:
        bigger_list = []
        less_list = []
        anchor_index = int(len(bbox_list) / 2)
        anchor_bbox = bbox_list[anchor_index]
        anchor_area = get_area_of_bbox(anchor_bbox, bbox_convention)
        for i in range(len(bbox_list)):
            if i == anchor_index:
                continue
            tmp_bbox = bbox_list[i]
            tmp_area = get_area_of_bbox(tmp_bbox, bbox_convention)
            if tmp_area >= anchor_area:
                bigger_list.append(tmp_bbox)
            else:
                less_list.append(tmp_bbox)
        if only_max:
            return qsort_bbox_list(bigger_list) + \
                [anchor_bbox, ] + less_list
        else:
            return qsort_bbox_list(bigger_list) + \
                [anchor_bbox, ] + qsort_bbox_list(less_list)


def sort_bbox_list_TTSH(img_height, 
                    img_width,
                    bbox_list: list,
                    only_max: bool = False,
                    bbox_convention: Literal['xyxy', 'xywh'] = 'xyxy'):
    """Sort a list of bboxes, by their area in pixel(W*H).

    Args:
        input_list (list):
            A list of bboxes. Each item is a list of (x1, y1, x2, y2)
        only_max (bool, optional):
            If True, only assure the max element at first place,
            others may not be well sorted.
            If False, return a well sorted descending list.
            Defaults to False.
        bbox_convention (str, optional):
            Bbox type, xyxy or xywh. Defaults to 'xyxy'.

    Returns:
        list:
            A sorted(maybe not so well) descending list.
    """
    # import pdb; pdb.set_trace()
    if len(bbox_list) <= 1:
        return bbox_list
    else:
        bigger_list = []
        less_list = []
        anchor_index = int(len(bbox_list) / 2)
        anchor_bbox = bbox_list[anchor_index]
        anchor_area = get_area_of_bbox(anchor_bbox, bbox_convention)
        anchor_position = get_position_of_bbox(img_height, img_width, anchor_bbox, bbox_convention)
        for i in range(len(bbox_list)):
            if i == anchor_index:
                continue
            tmp_bbox = bbox_list[i]
            tmp_area = get_area_of_bbox(tmp_bbox, bbox_convention)
            tmp_position = get_position_of_bbox(img_height, img_width, tmp_bbox, bbox_convention)
            if tmp_area * tmp_position >= anchor_area * anchor_position:
                bigger_list.append(tmp_bbox)
            else:
                less_list.append(tmp_bbox)
        if only_max:
            return sort_bbox_list_TTSH(img_height, img_width, bigger_list) + \
                [anchor_bbox, ] + less_list
        else:
            return sort_bbox_list_TTSH(img_height, img_width, bigger_list) + \
                [anchor_bbox, ] + sort_bbox_list_TTSH(img_height, img_width, less_list)


def get_area_of_bbox(
        bbox: Union[list, tuple],
        bbox_convention: Literal['xyxy', 'xywh'] = 'xyxy') -> float:
    """Get the area of a bbox_xyxy.

    Args:
        (Union[list, tuple]):
            A list of [x1, y1, x2, y2].
        bbox_convention (str, optional):
            Bbox type, xyxy or xywh. Defaults to 'xyxy'.

    Returns:
        float:
            Area of the bbox(|y2-y1|*|x2-x1|).
    """
    # import pdb;pdb.set_trace()
    if bbox_convention == 'xyxy':
        return abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
    elif bbox_convention == 'xywh':
        return abs(bbox[2] * bbox[3])
    else:
        raise TypeError(f'Wrong bbox convention: {bbox_convention}')


def get_position_of_bbox(img_height, 
        img_width,
        bbox: Union[list, tuple],
        bbox_convention: Literal['xyxy', 'xywh'] = 'xyxy') -> float:
    """Get the area of a bbox_xyxy.

    Args:
        (Union[list, tuple]):
            A list of [x1, y1, x2, y2].
        bbox_convention (str, optional):
            Bbox type, xyxy or xywh. Defaults to 'xyxy'.

    Returns:
        float:
            Area of the bbox(|y2-y1|*|x2-x1|).
    """
    # import pdb;pdb.set_trace()
    image_center_x = img_width / 2
    image_center_y = img_height / 2
    if bbox_convention == 'xyxy':
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        bbox_center_y = (bbox[1] + bbox[3]) / 2
        return 1 / (abs(image_center_x - bbox_center_x) + abs(image_center_y - bbox_center_y))
    elif bbox_convention == 'xywh':
        bbox_center_x = bbox[0]
        bbox_center_y = bbox[1]
        return 1 / (abs(image_center_x - bbox_center_x) + abs(image_center_y - bbox_center_y))
    else:
        raise TypeError(f'Wrong bbox convention: {bbox_convention}')


def calculate_iou(bbox1, bbox2):
    # Calculate the Intersection over Union (IoU) between two bounding boxes
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    
    union_area = bbox1_area + bbox2_area - intersection_area
    
    iou = intersection_area / union_area
    return iou


def non_max_suppression(bboxes, iou_threshold):
    # Sort the bounding boxes by their confidence scores (e.g., the probability of containing an object)
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    
    # Initialize a list to store the selected bounding boxes
    selected_bboxes = []
    
    # Perform non-maximum suppression
    while len(bboxes) > 0:
        current_bbox = bboxes[0]
        selected_bboxes.append(current_bbox)
        bboxes = bboxes[1:]
        
        remaining_bboxes = []
        for bbox in bboxes:
            iou = calculate_iou(current_bbox, bbox)
            if iou < iou_threshold:
                remaining_bboxes.append(bbox)
                
        bboxes = remaining_bboxes
        
    return selected_bboxes