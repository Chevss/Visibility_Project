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
edge_detection_mode = False
color_change_monitoring = False
distance_to_structure = 0.0  # Distance in meters

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

def get_edges(frame, bbox):
    """Apply Canny Edge Detection to the region inside the bounding box."""
    x1, y1, x2, y2 = bbox
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.sum(edges > 0)
    
    return edge_count, edges

def save_reference_values():
    """Save reference values to a file."""
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'boxes': bbox_list,
        'values': {k: {'rgb': v['rgb'], 'intensity': float(v['intensity']), 'distance': v['distance']} for k, v in reference_values.items()}
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

def compute_visibility(distance, intensity_change):
    """
    Compute visibility based on distance and intensity change.
    
    Args:
        distance: Distance from the camera to the structure in meters.
        intensity_change: The percentage change in intensity.
    
    Returns:
        Visibility in meters.
    """
    # Example formula: visibility decreases with intensity change
    # This is a placeholder formula and should be adjusted based on actual requirements
    visibility = distance / (1 + intensity_change / 100)
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
    global drawing, current_bbox, bbox_list, frame, distance_to_structure
    
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
        bbox_list.append((x1, y1, x2, y2))
        
        if frame is not None:
            rgb, intensity = get_average_colors(frame, (x1, y1, x2, y2))
            distance_to_structure = float(input(f"Enter the distance from the camera to the structure for box {len(bbox_list)} (in meters): "))
            reference_values[len(bbox_list)-1] = {
                'rgb': rgb.tolist(),
                'intensity': float(intensity),
                'distance': distance_to_structure
            }
            print(f"Box {len(bbox_list)} created with intensity: {intensity:.2f} and distance: {distance_to_structure} meters")
        
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
    global frame, monitoring, target_window, window_rect, edge_detection_mode, color_change_monitoring
    
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
    print("e: Toggle edge detection mode")
    print("c: Toggle color change monitoring")
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
                    distance = reference_values[i]['distance']
                    change_rgb = compute_rgb_change(rgb, ref_rgb)
                    change_intensity = compute_visibility_change(intensity, ref_intensity, 4.0)
                    
                    # Display status
                    color = (0, 255, 0) if change_intensity < 20 else (0, 0, 255)
                    status = f"Change: Intensity {change_intensity:.1f}%"
                    cv2.putText(frame, status, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if color_change_monitoring and change_rgb > 0:
                        # Compute visibility based on color change
                        visibility = compute_visibility(distance, change_intensity)
                        cv2.putText(frame, f"Visibility: {visibility:.1f}m", (x1, y2+40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                else:
                    # Show current intensity
                    cv2.putText(frame, f"I: {intensity:.1f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Only handle edge detection if in edge detection mode
                if edge_detection_mode:
                    edge_count, edges = get_edges(frame, bbox)
                    
                    # Calculate box dimensions
                    box_width = abs(x2 - x1)
                    box_height = abs(y2 - y1)
                    
                    # Ensure minimum size for visibility (at least 100x100)
                    display_width = max(box_width, 100)
                    display_height = max(box_height, 100)
                    
                    # Create or update edge window with the same size as the box
                    window_name = f"Edge Detection Box {i+1}"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name, edges)
                    
                    # Store window name in dictionary
                    edge_windows[i] = window_name
                    
                    # Resize the window to match the box dimensions
                    # Add a bit of extra padding (20px) for the window title bar
                    cv2.resizeWindow(window_name, display_width, display_height)
                    
                    # Display edge count on main window
                    cv2.putText(frame, f"Edges: {edge_count}", (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Add color monitoring if enabled
                if color_change_monitoring:
                    roi = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
                    if roi.size > 0:  # Ensure ROI is not empty
                        rgb_means = np.mean(roi, axis=(0, 1)).astype(int)
                        cv2.putText(frame, f"RGB: {rgb_means}", (x1, y2+20),
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
        elif key == ord('e'):
            # Toggle edge detection mode
            edge_detection_mode = not edge_detection_mode
            
            # If turning off edge detection, close all edge windows
            if not edge_detection_mode:
                close_all_edge_windows(edge_windows)
            
            # If switching from color monitoring to edge detection, clear boxes
            if edge_detection_mode and color_change_monitoring:
                color_change_monitoring = False
                bbox_list.clear()  # Clear boxes when switching modes
                reference_values.clear()
                print("Switched to Edge Detection Mode")
            else:
                print("Edge Detection Mode:", "Activated" if edge_detection_mode else "Deactivated")
                
        elif key == ord('c'):
            # Toggle color change monitoring
            color_change_monitoring = not color_change_monitoring
            
            # If turning on color monitoring, ensure edge detection is off
            if color_change_monitoring and edge_detection_mode:
                edge_detection_mode = False
                close_all_edge_windows(edge_windows)
                bbox_list.clear()  # Clear boxes when switching modes
                reference_values.clear()
                print("Switched to Color Change Monitoring Mode")
            else:
                print("Color Change Monitoring:", "Activated" if color_change_monitoring else "Deactivated")

    # Clean up all windows
    close_all_edge_windows(edge_windows)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()