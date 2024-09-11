import cv2
import zmq
import pickle
import zlib
import os
import numpy as np

def start_client():
    # Set ZeroMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://192.168.123.162:5555")  # Replace with the server's IP address
    print("The client is connected and waiting to receive data...")

    save_counter = 0

    try:
        while True:
            try:
                # Receive compressed color image data
                compressed_color_data = socket.recv()
                
                # Receive compressed depth image data
                compressed_depth_data = socket.recv()

                # Decompress the received data
                color_data = zlib.decompress(compressed_color_data)
                depth_data = zlib.decompress(compressed_depth_data)

                # Load the color image from the decompressed data
                color_frame_data = pickle.loads(color_data)
                color_frame = cv2.imdecode(color_frame_data, cv2.IMREAD_COLOR)

                # Convert the depth data back to a NumPy array and reshape it
                depth_frame = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)

                # Apply a colormap to the depth frame for visualization
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.05), cv2.COLORMAP_JET)

                # Display the color image and the depth colormap
                cv2.imshow('RGB Stream', color_frame)
                cv2.imshow('Depth Stream', depth_colormap)

                # Check for keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print("Save key 's' pressed.")
                    
                    # Save the RGB image as a PNG
                    rgb_filename = os.path.expanduser(f"~/aruco_images/rgb_image_{save_counter}.jpg")
                    cv2.imwrite(rgb_filename, color_frame)
                    print(f"Saved RGB image: {rgb_filename}")

                    # Save the depth colormap as a PNG
                    depth_png_filename = os.path.expanduser(f"~/aruco_images/depth_image_{save_counter}.png")
                    cv2.imwrite(depth_png_filename, depth_colormap)
                    print(f"Saved Depth image as PNG: {depth_png_filename}")

                    # Save the raw depth image as a NumPy array
                    depth_npy_filename = os.path.expanduser(f"~/aruco_images/depth_image_{save_counter}.npy")
                    np.save(depth_npy_filename, depth_frame)
                    print(f"Saved Depth image as NumPy array: {depth_npy_filename}")

                    # Increment the counter for the next save
                    save_counter += 1

            except Exception as e:
                print(f"An error occurred while receiving data: {e}")
                break
    finally:
        # Terminate the context and close any OpenCV windows
        context.term()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_client()
