import cv2
import math
import numpy as np
import torch
from torchvision import transforms

from internal.yolov7_pose.detect_pose import Y7Detect, draw_kpts, draw_boxes
from internal.yolo_pose import YoloPose, draw_kpts, draw_boxes
from internal.track_sort.Sort import SORT
from internal.classification_stgcn.Actionsrecognition.ActionsEstLoader import TSSTG
from internal.notification.telegram_sender import send_alarm_telegram


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def check_fall(previous_state, current_state, box):
    """
    :param previous_state: Last state of the person, 1: fall, 2: no fall
    :param current_state: Current state
    :param box: bounding box xyxy
    :return: true if person fall, else false
    """
    if current_state == 1:
        return True
    else:
        if previous_state == 1:
            x1, y1, x2, y2 = box
            ratio = (x2 - x1) / (y2 - y1)
            if ratio > 1:
                return True

    return False


class FallDetector(object):
    def __init__(self, yolo_weight_path, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.y7_pose = Y7Detect(yolo_weight_path)
        # self.yolo_pose = YoloPose(yolo_weight_path)
        self.action_model = TSSTG(device=device, skip=True)
        self.tracker = SORT(max_age=30, n_init=3, max_iou_distance=0.7)  # sort
        self.memory = {}

    def process_image(self, image):
        frame = image.copy()
        h, w, _ = frame.shape
        bbox, label, score, label_id, kpts = self.y7_pose.predict(frame)
        id_hold = []
        for i, box in enumerate(bbox):
            # check and remove bbox
            if box[0] < 10 or box[1] < 10 or box[2] > w - 10 or box[3] > h - 10:
                id_hold.append(False)
                continue
            id_hold.append(True)
        bbox, score, kpts = np.array(bbox), np.array(score), np.array(kpts)
        bbox, score, kpts = bbox[id_hold], score[id_hold], kpts[id_hold]

        # ***************************** TRACKING **************************************************
        if len(bbox) != 0:
            data = self.tracker.update(bbox, score, kpts, frame)
            for outputs in data:
                if len(outputs['bbox']) != 0:
                    box, score, kpt, track_id, list_kpt = (outputs['bbox'], outputs['score'], outputs['kpt'],
                                                           outputs['id'], outputs['list_kpt'])
                    kpt = kpt[:, :2].astype('int')

                    if str(track_id) not in self.memory:
                        self.memory.update(
                            {str(track_id): [str(track_id), 0, 0, 0]}
                            # track_id, count, fall state, sent alarm
                        )

                    # draw bounding boxes
                    # draw_boxes(frame, box, color=get_color((abs(int(track_id)))), label='Person', scores=score)
                    draw_boxes(frame, box, color=get_color((abs(int(track_id)))))

                    # draw person poses
                    draw_kpts(frame, [kpt])
                    color = (0, 255, 255)
                    previous_state = self.memory[str(track_id)][2]
                    current_state = 0
                    action_name = ''
                    # ************************************ PREDICT ACTION ********************************
                    if len(list_kpt) == 15:
                        # action, score = action_model.predict([list_kpt], w, h, batch_size=1)
                        action, score = self.action_model.predict(list_kpt, (w, h))
                        try:
                            if action[0] == "Fall Down":
                                current_state = 1
                                action_name = "Fall Down"
                            else:
                                current_state = 0
                                action_name = action[0]
                        except:
                            pass

                    check = check_fall(previous_state, current_state, box)
                    first_time = self.memory[str(track_id)][3]
                    if check:
                        action_name = "Fall Down"
                        self.memory[str(track_id)][2] = 1
                        color = (0, 0, 255)
                        x, y = int(w / 3), 70
                        font_scale = 2

                        text_size, _ = cv2.getTextSize('Person Fell Down', 0, font_scale, 2)
                        text_w, text_h = text_size
                        cv2.rectangle(frame, (x - 3, y - 3), (x + text_w + 6, y + text_h + 6), (0, 0, 0), -1)
                        cv2.putText(frame, 'Person Fell Down', (x, y + text_h + font_scale - 1), 0, font_scale, [0, 0, 255], thickness=2,
                                    lineType=cv2.LINE_AA)
                        if first_time == 0:
                            self.memory[str(track_id)][3] = 1
                            send_alarm_telegram(frame)

                    cv2.putText(frame, '{}'.format(action_name),
                                (max(box[0] + 5, 0), box[1] + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


            # update count memory with id track
            keys = list(self.memory.keys())
            for key in keys:
                if self.memory[key][1] > 100:
                    del self.memory[key]
                    continue
                self.memory.update({key: [self.memory[key][0], self.memory[key][1] + 1, self.memory[key][2], self.memory[key][3]]})

        return frame
