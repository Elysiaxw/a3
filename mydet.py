#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils
import argparse

img_path="/home/nvidia/jetson-inference/data/images/ap.jpeg"
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
img = jetson.utils.loadImage(img_path)
detections = net.Detection(img)
#display.Render(img)
#display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
jetson.utils.saveImage("/home/nvidia/jetson-inference/examples/myimage.jpg",img)
