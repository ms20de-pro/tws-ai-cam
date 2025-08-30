#!/usr/bin/python3

# MJPEG streaming server with detection overlay
# Auth: Matthias Schmidt
# Some parts copied from https://picamera.readthedocs.io/en/release-1.13/recipes2.html
# Run this script, then point a web browser at http:<this-ip-address>:8000
# Note: needs simplejpeg to be installed (pip3 install simplejpeg).


import io
import logging
import socketserver
import cv2
from http import server
from threading import Condition
import argparse
import paho.mqtt.client as mqtt
from picamera2 import Picamera2, CompletedRequest
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics)
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from typing import List
import sys
from time import time
import numpy as np

PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming for Timberwolf Server</title>
</head>
<body>
<h1>Picamera2 MJPEG Streaming for Timberwolf Server</h1>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""


# Detection result structure

class DetectionsBase:
    """
    Base class for detection results.
    """
    def __init__(self, category, conf, box):
        self.category = category
        self.conf = conf
        self.box = box

class Detection(DetectionsBase):
    """
    Class representing a single detection result.
    """
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        #box = imx500.convert_inference_coords(coords, metadata, picam2)

        # try to calculate the bounding box in the original image coordinates
        vh, vw = (args.width, args.height)

        # coords to box
        y1 = int(coords[0][0] * vw)
        x1 = int(coords[1][0] * vh)
        y2 = int(coords[2][0] * vw)
        x2 = int(coords[3][0] * vh)
        box = (x1, y1, x2, y2)

        logging.debug("Coords: {}".format(coords))
        logging.debug("Box:  {}".format(box))
        super().__init__(category, conf, box)

class RecentDetection(DetectionsBase):
    """
    Class representing a recent detection result.
    """
    def __init__(self, detection: Detection):
        super().__init__(detection.category, detection.conf, detection.box)
        self.updateLastSeen()

    def updateLastSeen(self):
        self.lastSeen = time()

    def updateBox(self, box):
        self.box = box

class Detections:
    def __init__(self):
        self.detections = []

    def add_or_update(self, detection: Detection) -> bool:
        """ Return True if the detection added as new, False if updated """
        def boxes_close(box1, box2, tol=5):
            # Compare each coordinate with tolerance
            return all(abs(a - b) <= tol for a, b in zip(box1, box2))

        # check if it has only one detection of this category
        if sum(det.category == detection.category for det in self.detections) == 1:
            # has this the detection been seen in the last second
            det = next(det for det in self.detections if det.category == detection.category)
            if det.lastSeen > time() - 1:
                logging.debug("Detection '%s' updated", detection.category)
                det.updateLastSeen()
                det.updateBox(detection.box)
                return False

        # for multiple detections find nearest box
        for i, det in enumerate(self.detections):
            if det.category == detection.category and boxes_close(det.box, detection.box, args.movement_threshold):
                det.updateLastSeen()
                det.updateBox(detection.box)
                return False
        
        self.detections.append(RecentDetection(detection))
        return True

    def clear(self):
        self.detections.clear()

    def cleanup(self, max_age=5):
        """ Remove detections not seen for more than max_age seconds """
        now = time()
        self.detections = [det for det in self.detections if now - det.lastSeen <= max_age]

    def getDetections(self):
        return self.detections

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            labels = get_labels()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                        detections = recent_detections.getDetections()

                        # send frame directly if there are no detections
                        frame_to_send = frame

                        if detections:
                            # Decode JPEG to image array
                            img_array = cv2.imdecode(
                                np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR
                            )
                            # Draw bounding boxes
                            for det in detections:
                                x1, y1, x2, y2 = det.box
                                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 1)

                                # Add text label
                                label = f"{labels[int(det.category)]}: {det.conf:.2f}"
                                cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Re-encode image to JPEG
                            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
                            ret, jpeg = cv2.imencode('.jpg', img_array, encode_param)
                            if not ret:
                                continue
                            frame_to_send = jpeg.tobytes()
                        
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame_to_send))
                        self.end_headers()
                        self.wfile.write(frame_to_send)
                        self.wfile.write(b'\r\n')
            except Exception as e:
                logging.info('Removed streaming client %s: %s', self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

recent_detections = Detections()
output = StreamingOutput()
intrinsics = None
LABELS = None
imx500 = None
args = None
picam2 = None

def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


# Parse detection results from IMX500 output
def parse_detection_results(request: CompletedRequest) -> List[Detection]:
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold

    metadata = request.get_metadata()
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return []
    
    _, input_h = imx500.get_input_size()

    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
    if bbox_normalization:
        boxes = boxes / input_h
    
    if bbox_order == "xy":
        boxes = boxes[:, [1, 0, 3, 2]]

    boxes = np.array_split(boxes, 4, axis=1)
    boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections



# Draw bounding boxes and publish detection results
def handle_detection_results(request: CompletedRequest):
    import json
    detections = parse_detection_results(request)
    if detections:
        labels = get_labels()
        # Draw on image
        output_detections = []
        for det in detections:
            if recent_detections.add_or_update(det):
                #x1, y1, x2, y2 = det.bbox
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{labels[int(det.category)]} ({det.conf:.2f} {det.box})"
                logging.info("Detected %s", label)
                #cv2.putText(frame, f"{det.label} {det.score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                output_detections.append(det)
            else:
                logging.debug("Updated existing detection: %s", labels[int(det.category)])
        # Overwrite the image in the request (for streaming)
        #request.set_image(frame)

        if output_detections:
            mqtt_client.publish("picamera2/detections", json.dumps(
                {
                    "change": "true",
                    "detections": [
                        {
                            "category_id": int(det.category),
                            "label": labels[int(det.category)],
                            "confidence": float(det.conf),
                            "bbox": [float(coord) for coord in det.box]
                        }
                        for det in detections
                    ]
                }
            ))




    recent_detections.cleanup()

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Picamera2 MJPEG streaming with MQTT")
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preprocess the image with preserve aspect ratio")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")

    parser.add_argument("--width", type=int, default=640, help="Width of the input image")
    parser.add_argument("--height", type=int, default=480, help="Height of the input image")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality (0-100)")
    parser.add_argument("--min-score", type=float, default=0.2, help="Minimum score threshold for detections")
    parser.add_argument("--movement-threshold", type=int, default=5, help="Minimum movement threshold for detections")

    # mqtt options
    parser.add_argument('--mqtt-host', required=True, help='MQTT broker host')
    parser.add_argument('--mqtt-user', required=False, help='MQTT username')
    parser.add_argument('--mqtt-password', required=False, help='MQTT password')

    # logging options
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')

    return parser.parse_args()

def run_app():
    global output, imx500, intrinsics, args, mqtt_client, picam2
    args = get_args()

    # set loglevel
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    # Set up MQTT client
    mqtt_client = mqtt.Client()
    if args.mqtt_password:
        mqtt_client.username_pw_set(args.mqtt_user, args.mqtt_password)

    mqtt_client.connect(args.mqtt_host)
    mqtt_client.loop_start()


    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={
            "size": (args.width, args.height)
        },
        controls={
            "FrameDurationLimits": (100000, 100000)
        }))
    imx500.show_network_fw_progress_bar()
    picam2.start_recording(JpegEncoder(q=args.jpeg_quality), FileOutput(output))

    if intrinsics.preserve_aspect_ratio:
        logging.debug("Preserving aspect ratio")
        #imx500.set_auto_aspect_ratio()

    picam2.pre_callback = handle_detection_results
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        picam2.stop_recording()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, style='{', format='{asctime} {levelname} {message}')
    logging.info("Starting MJPEG streaming server...")
    run_app()
    logging.info("Server stopped.")
