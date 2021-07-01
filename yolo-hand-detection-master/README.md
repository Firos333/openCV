# YOLO-Hand-Detection
Scene hand detection for real world images.

![Hand Detection Example](https://github.com/Firos333/openCV-Object-detection/blob/master/images/yolohand.png)

Pc Specifications :  4GB Ram, Lenovo G50. Intel Pentium Processor. DDr3 type RAM. 2.16Ghz 

### Idea
To detect hand gestures, we first have to detect the hand position in space. This pre-trained network is able to extract hands out of a `2D RGB` image, by using the YOLOv3 neural network.







```bash
# with python 3
python demo_webcam.py
```

Or this one to run a webcam detrector with YOLOv3 tiny:

```bash
# with python 3
python demo_webcam.py -n tiny
```

For Yolov3-Tiny-PRN use the following command:

```bash
# with python 3
python demo_webcam.py -n prn
```

### Download

- YOLOv3 Cross-Dataset
	- [Configuration](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.cfg)
	- [Weights](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights)
- YOLOv3-tiny Cross-Hands
	- [Configuration](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny.cfg)
	- [Weights](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny.weights)
- YOLOv3-tiny-prn Cross-Hands
	- [Configuration](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny-prn.cfg)
	- [Weights](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands-tiny-prn.weights)

If you are interested in the CMU Hand DB results, please check the [release](https://github.com/cansik/yolo-hand-detection/releases/tag/pretrained) section.


