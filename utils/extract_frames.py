import cv2
import glob
from pathlib import Path

from PIL import Image
import warnings
import os


def extract_frames(file: Path):
    """
    Extract frames from a video file and save to ./frames subdirectory

    Args:
        file (Path): path to the video file

    Returns:
        None
    """
    if not os.path.exists(f"../data/interim_data/frames/{file.name}"):
        os.makedirs(f"../data/interim_data/frames/{file.name}")
    vidcap = cv2.VideoCapture(str(file))
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(f"../data/interim_data/frames/{file.name}/frame%d.jpg" % count, image)     # save frame as JPEG file
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def extract_all_frames():
    for f in glob.glob("../data/raw_data/videos/*.mp4"):
        extract_frames(Path(f))


extract_all_frames()


def convert_frames_to_html(file: Path):
    """
    Convert a frame to html and save to ./frames_html subdirectory
    
    Args:
        filepath (str): path to the frame

    Returns:
        str: ascii art representation of the frame
    """
    ascii_art = AsciiArt.from_image(str(file))
    ascii_art.to_html_file(f'frames_html/{file.stem}.html')
    return ascii_art