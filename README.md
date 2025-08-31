# Overview

Use Raspberry Pi AI Camera with the Timberwolf Server VISU

This software runs an MJPEG streaming server on a Raspberry Pi camera using Picamera2 and IMX500 neural network hardware.
It performs real-time object detection, overlays bounding boxes and labels on the video stream, and publishes detection results via MQTT.

## Features

- Live MJPEG video streaming in the Timberwolf Server VISU
- Real-time object detection with bounding boxes and labels
- MQTT publishing of detection events with a lot of possibilities for Smart Home integration
- Configurable detection model, image size, and thresholds

## Usage

1. You need to have a Raspberry Pi with the AI camera module installed and the software for the camera set up. See https://www.raspberrypi.com/documentation/accessories/ai-camera.html for more information.
2. Install dependencies:
   - <code>pip3 install simplejpeg paho-mqtt opencv-python</code>
3. Run the script:
   - <code>python3 cam_mpeg.py --mqtt-host &lt;broker&gt; [other options]</code>. Due to the upload speed for the AI model, it can take some time until the stream is available.
   - Use systemd example to run the script as a service in the background and automatically start on boot.
4. Add the stream to camera widget in the Timberwolf Server VISU.

## Install as Systemd Service

Copy `cam_mpeg.py` and `assets` folder to a suitable location, e.g., `/usr/local/bin/ai-cam-tws`:

```bash
sudo cp cam_mpeg.py /usr/local/bin/ai-cam-tws
sudo cp -r assets /usr/local/bin/ai-cam-tws
```

Make the script executable:

```bash
sudo chmod +x /usr/local/bin/ai-cam-tws
```

Create an configuration file for the service:

```bash
sudo nano /etc/ai-cam-tws.conf
```

Copy the file provided in the `systemd` directory to `/etc/systemd/system/ai-cam-tws.service`:

```bash
sudo cp systemd/ai-cam-tws.service /etc/systemd/system/
```

Then, enable and start the service:

```bash
sudo systemctl enable ai-cam-tws
sudo systemctl start ai-cam-tws
```

Check the status of the service:

```bash
sudo systemctl status ai-cam-tws
```

Due to the upload speed for the AI model, it can take some time until the stream is available.

## Command-line options

- <code>--model</code>: Path to different neural network model (.rpk)
- <code>--width</code>, <code>--height</code>: Image size
- <code>--jpeg-quality</code>: JPEG compression quality
- <code>--threshold</code>: Detection confidence threshold
- <code>--mqtt-host</code>: MQTT broker host
- <code>--mqtt-user</code>, <code>--mqtt-password</code>: MQTT credentials

See <code>python3 cam_mpeg.py --help</code> for all options.

## MQTT Output

Detection results are published as JSON to <code>picamera2/detections</code> topic.


## Timberwolf Server Configuration

### VISU

Create a new camera widget in the Timberwolf Server VISU.

Use setting MJPEG Stream and enter the URL.
`http://<host-or-ip>:8000/stream.mjpg`

Allow capture if you want to trigger a capture via MQTT.
The last captures are stored in the Timberwolf Server and can be viewed in the detail view of the camera widget if enabled in the widget settings.

### MQTT Subsystem

- Open "MQTT Device Manager" Add a new device for the Raspberry Pi, with the "Main Level" <code>picamera2</code>.
- In this new device add an "App level Topic" with the URI <code>/detections</code> and format JSON.
- Subscribe to selector <code>change</code> with format boolean.
- This new boolean object will be triggered whenever a new detection is published and can be connected to trigger a capture in the Timberwolf Server VISU.


