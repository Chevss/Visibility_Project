import cv2
import numpy as np
from mss import mss
import win32gui
import win32con
import time
import json
from datetime import datetime

# Global variables
drawing = False
bbox_list = []
current_bbox = []
reference_values = {}
monitoring = False
frame = None
target_window = None
window_rect = None
color_change_monitoring = False
distance_to_structure = 0.0  # Distance in meters
background_color = np.array([0, 0, 0])  # Default background color (black)
setting_background_color = False

def get_average_colors(frame, bbox):
    """Get both RGB and grayscale intensity for a region."""
    x1, y1, x2, y2 = bbox
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    roi = frame[y:y+h, x:x+w]
    rgb_means = np.mean(roi, axis=(0, 1))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    intensity = np.mean(gray)
    
    return rgb_means, intensity

def get_edges(frame, bbox, threshold1=100, threshold2=200):
    """Apply Canny Edge Detection to the region inside the bounding box."""
    x1, y1, x2, y2 = bbox
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    edge_count = np.sum(edges > 0)
    
    return edge_count, edges

def save_reference_values():
    """Save reference values to a file."""
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'boxes': bbox_list,
        'values': {k: {'rgb': v['rgb'], 'intensity': float(v['intensity']), 'distance': float(v['distance']), 'edges': int(v['edges'])} for k, v in reference_values.items()}
    }
    with open('reference_values.json', 'w') as f:
        json.dump(data, f)
    print("Reference values saved")

def compute_visibility_change(current, reference, threshold=0.5):
    """
    Compute visibility change from reference.
    
    Args:
        current: Current intensity value
        reference: Reference intensity value
        threshold: Minimum change percentage to consider significant (default 0.5%)
    
    Returns:
        The percentage change if above threshold, otherwise 0
    """
    if abs(reference) < 1e-6:
        return 0
    
    change = abs((current - reference) / reference) * 100
    
    # Return 0 if the change is below the threshold
    if change < threshold:
        return 0
        
    return change

def compute_visibility(distance, intensity_change, edge_change, color_change):
    """
    Compute visibility based on distance, intensity change, edge change, and color change.
    
    Args:
        distance: Distance from the camera to the structure in meters.
        intensity_change: The percentage change in intensity.
        edge_change: The percentage change in edge count.
        color_change: The percentage change in color.
    
    Returns:
        Visibility in meters.
    """
    # Example formula: visibility decreases with intensity, edge, and color change
    # This is a placeholder formula and should be adjusted based on actual requirements
    visibility = distance / (1 + (intensity_change + edge_change + color_change) / 300)
    return visibility

def compute_rgb_change(current_rgb, reference_rgb, threshold=5.0):
    """
    Compute RGB change from reference.
    
    Args:
        current_rgb: Current RGB values
        reference_rgb: Reference RGB values
        threshold: Minimum change percentage to consider significant (default 5%)
    
    Returns:
        The percentage change if above threshold, otherwise 0
    """
    change = np.linalg.norm(current_rgb - reference_rgb) / np.linalg.norm(reference_rgb) * 100
    
    # Return 0 if the change is below the threshold
    if change < threshold:
        return 0
        
    return change

def compute_color_similarity(current_rgb, background_rgb):
    """
    Compute color similarity to the background color.
    
    Args:
        current_rgb: Current RGB values
        background_rgb: Background RGB values
    
    Returns:
        The similarity percentage (0% to 100%).
    """
    similarity = np.linalg.norm(current_rgb - background_rgb) / np.linalg.norm(background_rgb) * 100
    return similarity

def compute_visibility_percentage(intensity_change, edge_change, color_change):
    """
    Compute visibility percentage based on intensity change, edge change, and color change.
    
    Args:
        intensity_change: The percentage change in intensity.
        edge_change: The percentage change in edge count.
        color_change: The percentage change in color.
    
    Returns:
        Visibility percentage (100% to 0%).
    """
    # Example formula: visibility decreases with intensity, edge, and color change
    # This is a placeholder formula and should be adjusted based on actual requirements
    visibility_percentage = max(0, 100 - (intensity_change + edge_change + color_change) / 3)
    return visibility_percentage

def get_window_by_title(title_pattern):
    """Find window by partial title match."""
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            if title_pattern.lower() in window_text.lower():
                windows.append((hwnd, window_text))
        return True
    
    windows = []
    win32gui.EnumWindows(callback, windows)
    return windows

def set_window_position():
    """Set the target window position."""
    global window_rect
    if target_window:
        try:
            window_rect = win32gui.GetWindowRect(target_window)
        except Exception as e:
            print(f"Error getting window position: {e}")
            return False
    return True

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for drawing boxes."""
    global drawing, current_bbox, bbox_list, frame, distance_to_structure, setting_background_color, background_color
    
    if event == cv2.EVENT_LBUTTONDOWN and not monitoring:
        drawing = True
        current_bbox = [(x, y)]
    
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if frame is not None:
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, current_bbox[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Visibility Monitor", temp_frame)
    
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        current_bbox.append((x, y))
        x1, y1 = current_bbox[0]
        x2, y2 = current_bbox[1]
        
        if setting_background_color:
            background_color, _ = get_average_colors(frame, (x1, y1, x2, y2))
            print(f"Background color set to: {background_color}")
            setting_background_color = False
        else:
            bbox_list.append((x1, y1, x2, y2))
            
            if frame is not None:
                rgb, intensity = get_average_colors(frame, (x1, y1, x2, y2))
                edge_count, _ = get_edges(frame, (x1, y1, x2, y2), threshold1=50, threshold2=150)
                distance_to_structure = float(input(f"Enter the distance from the camera to the structure for box {len(bbox_list)} (in meters): "))
                reference_values[len(bbox_list)-1] = {
                    'rgb': rgb.tolist(),
                    'intensity': float(intensity),
                    'distance': distance_to_structure,
                    'edges': edge_count
                }
                print(f"Box {len(bbox_list)} created with intensity: {intensity:.2f}, edges: {edge_count}, and distance: {distance_to_structure} meters")
        
        current_bbox = []

def close_all_edge_windows(edge_windows):
    """Close all edge detection windows."""
    for window_name in edge_windows.values():
        try:
            cv2.destroyWindow(window_name)
        except:
            pass
    edge_windows.clear()

def main():
    global frame, monitoring, target_window, window_rect, color_change_monitoring, background_color, setting_background_color
    
    # Initialize screen capture
    sct = mss()

    # Find Edge window
    windows = get_window_by_title("edge")
    if not windows:
        print("No Edge window found! Please open Edge browser first.")
        return
    
    # Let user select which window if multiple found
    if len(windows) > 1:
        print("Multiple Edge windows found. Please select one:")
        for i, (_, title) in enumerate(windows):
            print(f"{i}: {title}")
        selection = int(input("Enter number: "))
        target_window = windows[selection][0]
    else:
        target_window = windows[0][0]

    # Create window
    cv2.namedWindow("Visibility Monitor")
    cv2.setMouseCallback("Visibility Monitor", mouse_callback)
    
    print("\nControls:")
    print("Click and drag to create boxes")
    print("m: Start/stop monitoring")
    print("r: Reset boxes")
    print("s: Save reference values")
    print("c: Toggle color change monitoring")
    print("b: Set background color")
    print("q: Quit\n")
    
    # Dictionary to store edge windows and their information
    edge_windows = {}
    
    while True:
        if not set_window_position() or not window_rect:
            print("Waiting for window...")
            time.sleep(1)
            continue

        try:
            # Capture specific window
            screenshot = sct.grab({
                'left': window_rect[0],
                'top': window_rect[1],
                'width': window_rect[2] - window_rect[0],
                'height': window_rect[3] - window_rect[1]
            })
            
            frame = np.array(screenshot)
            if frame.size == 0:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Draw boxes and show measurements
            for i, bbox in enumerate(bbox_list):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Get current values
                rgb, intensity = get_average_colors(frame, bbox)
                
                if monitoring and i in reference_values:
                    # Compare with reference
                    ref_rgb = np.array(reference_values[i]['rgb'])
                    ref_intensity = reference_values[i]['intensity']
                    ref_edges = reference_values[i]['edges']
                    distance = reference_values[i]['distance']
                    change_rgb = compute_rgb_change(rgb, ref_rgb)
                    change_intensity = compute_visibility_change(intensity, ref_intensity, 4.0)
                    edge_count, _ = get_edges(frame, bbox, threshold1=50, threshold2=150)
                    change_edges = compute_visibility_change(edge_count, ref_edges, 4.0)
                    color_similarity = compute_color_similarity(rgb, background_color)
                    
                    # Display status
                    color = (0, 255, 0) if change_intensity < 20 else (0, 0, 255)
                    status = f"Change: {change_intensity:.1f}%"
                    cv2.putText(frame, status, (x1, y1-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if color_change_monitoring and change_rgb > 0:
                        # Compute visibility percentage based on color change, edge change, and color similarity
                        visibility_percentage = compute_visibility_percentage(change_intensity, change_edges, color_similarity)
                        cv2.putText(frame, f"Distance: {distance:.1f}m", (x1, y2+30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(frame, f"V Ratio: {visibility_percentage:.1f}%", (x1, y2+50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        # Display edge count
                        cv2.putText(frame, f"Edges: {edge_count}", (x1, y2+70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    # Show current intensity
                    cv2.putText(frame, f"I: {intensity:.1f}", (x1, y1-40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Add color monitoring if enabled
                if color_change_monitoring:
                    roi = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
                    if roi.size > 0:  # Ensure ROI is not empty
                        rgb_means = np.mean(roi, axis=(0, 1)).astype(int)
                        cv2.putText(frame, f"{rgb_means}", (x1, y2+10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow("Visibility Monitor", frame)

        except Exception as e:
            print(f"Error capturing window: {e}")
            time.sleep(1)
            continue
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            monitoring = not monitoring
            print("Monitoring:", "Started" if monitoring else "Stopped")
        elif key == ord('r'):
            # Close all edge windows before clearing boxes
            close_all_edge_windows(edge_windows)
            bbox_list.clear()
            reference_values.clear()
            monitoring = False
            print("Reset complete")
        elif key == ord('s'):
            save_reference_values()
        elif key == ord('c'):
            # Toggle color change monitoring
            color_change_monitoring = not color_change_monitoring
            print("Color Change Monitoring:", "Activated" if color_change_monitoring else "Deactivated")
        elif key == ord('b'):
            # Set background color
            setting_background_color = True
            print("Draw a box to set the background color")

    # Clean up all windows
    close_all_edge_windows(edge_windows)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()