
import argparse
import os
import time
import torch
import cv2


from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
from subprocess import Popen
from PIL import Image

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template('index.html')


def get_frame():
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(
        folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(
        os.path.join(folder_path, x)))
    filename = predict_img.imgpath
    image_path = folder_path+'/'+latest_subfolder+'/'+filename
    video = cv2.VideoCapture(image_path)  # detected video path
    # video = cv2.VideoCapture("video.mp4")
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        # control the frame rate to display one frame every 100 milliseconds:
        time.sleep(0.1)


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(
        folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(
        os.path.join(folder_path, x)))
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ", directory)
    filename = predict_img.imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    # print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension == 'jpg':
        return send_from_directory(directory, filename, environ)

    elif file_extension == 'mp4':
        return render_template('index.html')

    else:
        return "Invalid file format"


@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)

            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()
            if file_extension == 'jpg':
                process = Popen(["python", "detect.py", '--source', filepath,
                                "--weights", "yolov7_training.pt"], shell=True)
                process.wait()

            elif file_extension == 'mp4':
                process = Popen(["python", "detect.py", '--source', filepath,
                                "--weights", "yolov7_training.pt"], shell=True)
                process.communicate()
                process.wait()

    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(
        folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(
        os.path.join(folder_path, x)))
    image_path = folder_path+'/'+latest_subfolder+'/'+f.filename
    return render_template('index.html', image_path=image_path)
    # return "done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask app exposing yolov7 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = torch.hub.load('.', 'custom', 'yolov7_training.pt', source='local')
    model.eval()
    # debug=True causes Restarting with stat
    app.run(host="0.0.0.0", port=args.port)
