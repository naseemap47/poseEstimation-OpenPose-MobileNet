import cv2

# Load Weights
net = cv2.dnn.readNetFromTensorflow('graph_opt.pb')

input_width = 368
input_height = 368
threshold = 0.2
DRAW = True


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_height, img_width, channel = img.shape
    
    net.setInput(cv2.dnn.blobFromImage(img, 1.0, (input_width, input_height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()

    # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    out = out[:, :19, :, :]

    assert(len(BODY_PARTS) == out.shape[1])

    points = []

    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heat_map = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heat_map)
        x = (img_width * point[0]) / out.shape[3]
        y = (img_height * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > threshold else None)

    final_points = []

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            final_points.append([points[idFrom], points[idTo]])
            if DRAW:
                cv2.line(img, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.circle(img, points[idFrom], 3, (0, 255, 255), cv2.FILLED)
                cv2.circle(img, points[idTo], 3, (0, 255, 255), cv2.FILLED)

    # for p in points:
    #     cv2.circle(
    #         img, p[0], 3,
    #         (255, 0, 255), cv2.FILLED
    #     )
    #     cv2.circle(
    #         img, p[1], 3,
    #         (0, 0, 255), cv2.FILLED
    #     )

    print(final_points)
    
    cv2.imshow('Pose Cam', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        cv2.destroyAllWindows()
        break

