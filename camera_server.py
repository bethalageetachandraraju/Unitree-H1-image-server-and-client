import pyrealsense2 as rs
import cv2
import numpy as np
import zmq
import pickle
import zlib

def start_server():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream different formats and resolutions
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Set ZeroMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind("tcp://*:5555")
    print("The server has started, waiting for client connections...")

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Encoding the images separately
            _, encoded_color_image = cv2.imencode('.jpg', color_image)
            _, encoded_depth_image = cv2.imencode('.jpg', depth_image)

            # Compressing data using pickle and zlib
            color_data = pickle.dumps(encoded_color_image)
            depth_data = pickle.dumps(encoded_depth_image)
            compressed_color_data = zlib.compress(color_data)
            compressed_depth_data = zlib.compress(depth_data)

            # Send the color image first, followed by the depth image
            socket.send(compressed_color_data, zmq.SNDMORE)
            socket.send(compressed_depth_data)

    finally:
        # Stop streaming and terminate context
        pipeline.stop()
        context.term()

if __name__ == "__main__":
    start_server()
