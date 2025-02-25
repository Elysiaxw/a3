#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutputimport jetson.inference
import jetson.utils

image_path="/home/nvidia/jetson-inference/data/images/ap.jpeg"
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)   
img = jetson.utils.loadImage(image_path) 
detections = net.Detect(img)
display.Render(img)
display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
