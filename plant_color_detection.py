import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Set dark mode style for the plot
plt.style.use('dark_background')

# RTSP URL for the webcam
RTSP_URL = "rtsp://buth:4ytkfe@192.168.1.210/live/ch00_1"

# Video Parameters
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

# Initialize the camera using RTSP URL
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Define the color ranges for detection in HSV
color_ranges = {
    'yellow': ((20, 100, 100), (30, 255, 255)),  # Lower and upper range for yellow
    'brown': ((10, 100, 20), (20, 255, 200)),  # Lower and upper range for brown
    'white': ((0, 0, 200), (180, 20, 255)),  # Lower and upper range for white
    'blue': ((90, 50, 50), (130, 255, 255))  # Lower and upper range for blue
}

# Define thresholds for each color
thresholds = {
    'yellow': 0.1,
    'brown': 0.1,
    'white': 0.02,
    'blue': 0.2
}

# Initialize deque to store color counts
color_counts = {color: deque(maxlen=100) for color in color_ranges}
frame_buffer = deque(maxlen=100)  # Initialize frame buffer

# Initialize plot
plt.ion()
fig, ax = plt.subplots()
lines = {
    color: ax.plot([], [], label=color, color=color)[0]
    for color in color_ranges
}
ax.legend()
ax.set_xlim(0, 100)
ax.set_ylim(0, 0.5)  # Set y-axis limit to 0.5 for normalized values

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_buffer.append(frame)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_frame, lower, upper)
        color_count = cv2.countNonZero(mask)
        normalized_count = color_count / (frame.shape[0] * frame.shape[1])  # Normalize the count
        color_counts[color].append(normalized_count)
        
        # Check if the normalized count exceeds the threshold
        if normalized_count > thresholds[color]:
            print(f"Alert: {color} count exceeded threshold with value {normalized_count:.2f}")
        
        # Update plot data
        lines[color].set_xdata(range(len(color_counts[color])))
        lines[color].set_ydata(color_counts[color])
    
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
