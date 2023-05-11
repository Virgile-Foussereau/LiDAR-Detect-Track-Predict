import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# read the png files
dir1 = os.listdir("PNG")
if len(dir1) != 0:
    imgs = ["PNG/arrow"+ str(i) + ".png" for i in range(1, 99)]
    frames = []
    fig = plt.figure()
    for i in imgs:
        new_frame = plt.imread(i)
        frames.append([plt.imshow(new_frame, animated=True)])

    # create animation using the png files
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat=False)
    plt.show()

    # save the animation as a mp4. This requires ffmpeg to be installed
    ani.save('animations/animationClustering_learning.mp4', writer='ffmpeg', fps=10)
    
dir2 = os.listdir("PNG2")
if len(dir2) != 0:
    imgs = ["PNG2/arrow"+ str(i) + ".png" for i in range(1, 99)]
    frames = []
    fig = plt.figure()
    for i in imgs:
        new_frame = plt.imread(i)
        frames.append([plt.imshow(new_frame, animated=True)])

    # create animation using the png files
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat=False)
    plt.show()

    # save the animation as a mp4. This requires ffmpeg to be installed
    ani.save('animations/animationClustering_geometry.mp4', writer='ffmpeg', fps=10)