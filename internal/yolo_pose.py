import time

from ultralytics import YOLO
import cv2
import random


def draw_boxes(image, boxes, label=None, scores=None, color=None):
    if color is None:
        color = (0, 255, 0)
    xmin, ymin, xmax, ymax = boxes
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    if scores is not None:
        cv2.putText(image, label + "-{:.2f}".format(scores), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image, boxes


def draw_kpts(image, kpts):
    for kpt in kpts:
        line_head = [6, 2, 1, 5]
        line_0 = [10, 8, 6, 5, 7, 9]
        line_1 = [12, 6, 5, 11]
        line_2 = [15, 13, 11, 12, 14, 16]

        kpt_head = kpt[line_head]
        kpt_head = kpt_head[(kpt_head != [0, 0]).all(axis=1)]

        kpt_line_0 = kpt[line_0]
        kpt_line_0 = kpt_line_0[(kpt_line_0 != [0, 0]).all(axis=1)]

        kpt_line_1 = kpt[line_1]
        kpt_line_1 = kpt_line_1[(kpt_line_1 != [0, 0]).all(axis=1)]

        kpt_line_2 = kpt[line_2]
        kpt_line_2 = kpt_line_2[(kpt_line_2 != [0, 0]).all(axis=1)]

        cv2.polylines(image, [kpt_head], False, (0, 255, 20), 5)
        cv2.polylines(image, [kpt_line_0], False, (255, 255, 20), 5)
        cv2.polylines(image, [kpt_line_1], True, (255, 20, 255), 5)
        cv2.polylines(image, [kpt_line_2], False, (20, 255, 255), 5)
        for idx, point in enumerate(kpt[5:]):
            cv2.circle(image, (point[0], point[1]), 5, (0, 0, 255), -1)
    return image


class YoloPose:
    def __init__(self, weights='yolo11n-pose.pt'):
        """
        params weights: 'yolo11n-pose.pt'
        """
        self.weights = weights
        self.model_image_size = 640  # 960
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        # Load a model
        self.model = YOLO(self.weights)  # load an official model
        self.class_names = self.model.names

        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]

    def predict(self, image_rgb):
        result = self.model(image_rgb, verbose=False)[0]


        kpts = result.keypoints.data.cpu().detach().numpy().astype(int)
        bboxes = result.boxes.xyxy.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().detach().numpy()
        lables_id = result.boxes.cls.cpu().detach().numpy()
        labels = [self.class_names[lid] for lid in lables_id]

        return bboxes, labels, scores, lables_id, kpts


def test():
    # Load a model
    model = YOLO("yolo11n-pose.pt")  # load an official model
    # model.export(format="openvino")  # creates 'yolov8n_openvino_model/'
    # Load the exported OpenVINO model
    # ov_model = YOLO("yolo11n-pose_openvino_model/")
    class_names = model.names
    path = '/home/tungpt37/Workspace/frame_00001.png'
    img = cv2.imread(path)
    results = model(img)
    results = model(img)
    start = time.time()
    results = model(img)
    end = time.time()
    print(end - start)
    for result in results:
        # print(result)
        kpts = result.keypoints.data.cpu().detach().numpy().astype(int)
        bboxes = result.boxes.xyxy.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().detach().numpy()
        lables_id = result.boxes.cls.cpu().detach().numpy()
        labels = [class_names[lid] for lid in lables_id]
        print(kpts)
        print(kpts.shape)
        print(bboxes.shape)
        draw_img = draw_kpts(img, kpts)
        cv2.imshow('img', draw_img)
        cv2.waitKey(0)
        # print(a)
        result.show()

    # model = YoloPose()
    # path = '/home/tungpt37/Workspace/frame_00001.png'
    # img = cv2.imread(path)
    # bboxes, labels, scores, lables_id, kpts = model.predict(img)
    # draw_img = draw_kpts(img, kpts)
    # cv2.imshow('img', draw_img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    test()