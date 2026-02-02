import sys
import pyzed.sl as sl
import cv2
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Convert ZED SVO2 file to MP4")
    parser.add_argument("input_svo", type=str, help="Path to the input .svo or .svo2 file")
    parser.add_argument("output_mp4", type=str, help="Path to the output .mp4 file")
    args = parser.parse_args()

    input_path = args.input_svo
    output_path = args.output_mp4

    if not os.path.isfile(input_path):
        print(f"Error: File not found {input_path}")
        sys.exit(1)

    # CREATE ZED CAMERA OBJECT
    cam = sl.Camera()

    # INIT PARAMETERS
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(input_path)
    init_params.svo_real_time_mode = False  # Process as fast as possible
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    # OPEN ZED CAMERA
    print(f"Opening SVO file: {input_path}")
    status = cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening SVO file: {status}")
        sys.exit(1)

    # GET CAMERA INFORMATION
    cam_info = cam.get_camera_information()
    fps = cam_info.camera_configuration.fps
    width = cam_info.camera_configuration.resolution.width
    height = cam_info.camera_configuration.resolution.height
    
    # SETUP OPENCV VIDEO WRITER
    # mp4v is a standard codec for mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print("Error opening video writer.")
        cam.close()
        sys.exit(1)

    print(f"Converting to {output_path} ({width}x{height} @ {fps}fps)...")

    # IMAGE CONTAINER
    mat = sl.Mat()
    
    # RUNTIME PARAMETERS
    runtime_parameters = sl.RuntimeParameters()

    frames_processed = 0
    total_frames = cam.get_svo_number_of_frames()

    while True:
        err = cam.grab(runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            # Retrieve the Left image (standard video view)
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            
            # Convert sl.Mat to numpy array (BGRA by default from SDK usually, need to check)
            # The get_data() returns numpy array (usually 4 channel BGRA or RGBA)
            frame_buffer = mat.get_data()

            # ZED SDK images are 4-channel (BGRA), OpenCV VideoWriter expects 3-channel (BGR)
            # We just need to drop the alpha channel.
            # However, check color format. Usually ZED numpy is BGRA.
            frame_bgr = cv2.cvtColor(frame_buffer, cv2.COLOR_BGRA2BGR)
            
            video_writer.write(frame_bgr)
            
            frames_processed += 1
            if frames_processed % 100 == 0:
                print(f"Processed {frames_processed}/{total_frames} frames...", end='\r')
        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("\nEnd of SVO file reached.")
            break
        else:
            print(f"\nError during grabbing: {err}")
            break

    # CLEANUP
    video_writer.release()
    cam.close()
    print("Conversion finished.")

if __name__ == "__main__":
    main()
