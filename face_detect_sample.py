# Sample face detection/alignment code
from utils import load_image
from align import AlignDlib

# Load sample image
img = load_image("unknown/unknown5.jpg")

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

# Get all face bounding boxes and thumbnails
face_bbs, thumbnails = alignment.getAllFaceBoundingBoxesAndThumbnails(img)

print("Number of face bounding boxes = {}".format(len(face_bbs)))
print("Number of face thumbnails = {}".format(len(thumbnails)))

