
# Facial recognition in videos

# This requires face_reco_base.py to be run first

import cv2
import matplotlib.pyplot as plt
import imageio
import pylab

CREATE_ANIMATED_GIF = True
CREATE_MP4_VIDEO  = False
DISPLAY_SAMPLE_FRAMES = False
TRAIN_WITH_ALL_SAMPLES = True

if TRAIN_WITH_ALL_SAMPLES == True:
    encoder,knn,svc,test_idx,targets = train_images(metadata2, embedded2, train_with_all_samples=True)


# Read source video file
filename = 'videos/retail_video1.mp4'

vid = imageio.get_reader(filename,'ffmpeg')

if DISPLAY_SAMPLE_FRAMES == True:
    nums = [100,150]
    
    # Display sample images from video
    for num in nums:
        image = vid.get_data(num)
        fig = pylab.figure()
        fig.suptitle('image #{}'.format(num), fontsize=20)
        pylab.imshow(image)
    pylab.show()

# Tag video frames with face labels 
def label_image(svc, knn, example_image, meta, embed):
    face_bbs, identities, meta, embed = identify_image_faces(example_image, svc, knn, encoder, meta, embed)
    img = label_cv2_image_faces(example_image, face_bbs, identities)
    # Convert cv2 RBG back to RGB format
    img = img[:,:,::-1]        
    return img, meta, embed

# Temp file for image labeling
temp_file = "temp.jpg"


if CREATE_ANIMATED_GIF == True:
    # Extract video frames for animated GIF
    frame_interval_secs = 1
    fps = vid.get_meta_data()["fps"]
    frame_interval_frames = int(frame_interval_secs * fps)
    
    num_frames = len(vid)
    frames = [i for i in range(0, num_frames, frame_interval_frames)]
    
    video_images = []
    
    for frame in frames:
        image = vid.get_data(frame)
        video_images.append(np.array(image))
    
    
    labeled_images = []
    
    for i, video_image in enumerate(video_images):
        print("Processing {} of {} video frames".format(i+1, len(video_images)))
        # TODO: Figure out how to do in-memory transform instead of using temp file
        imageio.imwrite(temp_file, video_image)
        video_image2 = load_image(temp_file)
        labeled_image, metadata2, embedded2 = label_image(svc, knn, video_image2, metadata2, embedded2)
        labeled_images.append(labeled_image)
    
    # Create animated GIF
    playback_frame_duration_secs=1
    
    print("Creating animated GIF...")
    
    with imageio.get_writer('movie.gif', mode='I', duration=playback_frame_duration_secs) as writer:
        for image in labeled_images:
            writer.append_data(image)
    
    print("Created animated GIF")


if CREATE_MP4_VIDEO == True:
    # Tag video frames with face labels for MP4 video
    
    vidnew=[]
    for i, image in enumerate (vid):
        
        ## label faces in video frame
        print("Processing {} of {} video frames".format(i+1, len(vid)))
        # TODO: Figure out how to do in-memory transform instead of using temp file
        imageio.imwrite(temp_file, image)
        video_image2 = load_image(temp_file)
        labeled_image = label_image(svc, knn, video_image2)
        
        #r = np.random.randint(-10,10,2)
        #n = cv2.rectangle(image,(600+r[0],400+r[1]),(700+r[0],300+r[1]),(0,255,0),3)
        
        ## append facial recognition return image to new list
        vidnew.append(labeled_image)
        
    # Create MP4 video
    writer = imageio.get_writer('newvideoname.mp4', fps=24)
    
    for im in vidnew:
        writer.append_data(im)
    writer.close()
