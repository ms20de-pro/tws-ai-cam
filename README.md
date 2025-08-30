# Overview

Use Raspberry Pi AI Camera with the Timberwolf Server VISU

This software runs an MJPEG streaming server on a Raspberry Pi camera using Picamera2 and IMX500 neural network hardware.
It performs real-time object detection, overlays bounding boxes and labels on the video stream, and publishes detection results via MQTT.

## Features

- Live MJPEG video streaming in browser
- Real-time object detection with bounding boxes and labels
- MQTT publishing of detection events
- Configurable detection model, image size, and thresholds

## Usage

1. Install dependencies:
   - <code>pip3 install simplejpeg paho-mqtt opencv-python</code>
2. Run the script:
   - <code>python3 cam_mpeg.py --mqtt-host &lt;broker&gt; [other options]</code>
   - Use systemd example to run the script as a service.
3. Add the stream to camera widget in the Timberwolf Server VISU.

## Command-line options

- <code>--model</code>: Path to neural network model (.rpk)
- <code>--width</code>, <code>--height</code>: Image size
- <code>--jpeg-quality</code>: JPEG compression quality
- <code>--threshold</code>: Detection confidence threshold
- <code>--mqtt-host</code>: MQTT broker host
- <code>--mqtt-user</code>, <code>--mqtt-password</code>: MQTT credentials

See <code>python3 cam_mpeg.py --help</code> for all options.

## MQTT Output

Detection results are published as JSON to <code>picamera2/detections</code> topic.


## Timberwolf Server Configuration

### VISU camera stream

Use setting MJPEG Stream and enter the URL.
`http://<host-or-ip>:8000/stream.mjpg`

### MQTT Subsystem

- Open "MQTT Device Manager" Add a new device for the Raspberry Pi, with the "Main Level" <code>picamera2</code>.
- In this new device add an "App level Topic" with the URI <code>/detections</code> and format JSON.
- Subscribe to selector <code>change</code> with format boolean.
- This new boolean object will be triggered whenever a new detection is published and can be connected to trigger a capture in the Timberwolf Server VISU.


