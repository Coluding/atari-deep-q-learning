import matplotlib.pyplot as plt

def render_all_frames_sequence(frames, title=""):
    plt.figure(figsize=(10, 10))
    for i, frame in enumerate(frames):
        plt.subplot(1, len(frames), i + 1)
        plt.imshow(frame.numpy())
        plt.axis('off')
    plt.suptitle(title)
    plt.show()