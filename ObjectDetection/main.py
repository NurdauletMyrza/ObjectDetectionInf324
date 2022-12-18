import cv2


net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

objects = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
           'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
           'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']
print(objects)

objects_need_to_find = ['person', 'car', 'bird', 'dog', 'horse', 'kite']
length = len(objects_need_to_find)


def detection_video(path, objects_to_find=objects):
    cap = cv2.VideoCapture(path)
    count = 1000

    while True:
        ret, frame = cap.read()
        if count < 0:
            break
        count -= 1
        (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            class_name = objects[class_id]
            try:
                index = objects_to_find.index(class_name)
                color = 255 * index // length
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (77, 77, 77), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (color, color, color), 5)
            except ValueError:
                print("", end='')

        cv2.imshow("Video", frame)

        if not ret or cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def detection_image(path, objects_to_find=objects):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    image = cv2.resize(image, (1024, 720), interpolation=cv2.INTER_AREA)

    (class_ids, scores, bboxes) = model.detect(image, confThreshold=0.3, nmsThreshold=.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = objects[class_id]
        try:
            index = objects_to_find.index(class_name)
            color = 255 * index // length
            cv2.putText(image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (color, color, color), 1)
            cv2.rectangle(image, (x, y), (x + w, y + h), (color, color, color), 5)
        except ValueError:
            print("", end='')

    cv2.imshow(path, image)

    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()


detection_image("./img1.png")
detection_image("./img2.png", ['car'])
detection_image("./img3.png")
detection_image("./img4.png")
detection_image("./img5.png")
detection_image("./img6.png")
detection_image("./img7.png")
detection_video("./SomeVideo.mp4")
detection_video("./Camera Highway.mp4")