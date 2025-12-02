import os
import cv2
import time
import traceback
import mediapipe as mp

from utils import draw_landmarks_on_image
from mediapipe.tasks.python import vision

# TODO
# Change global variables to function parameters
latest_annotated_bgr = None
latest_timestamp = -1


def annotate_frame(result, output_image: mp.Image, timestamp_ms: int):
    """
    Callback invoked by MediaPipe for each processed frame in LIVE_STREAM mode.
    - result: HandLandmarkerResult (hand_landmarks, world_landmarks, handedness)
    - output_image: mp.Image (RGB) associated with this result (can be resized if needed)
    - timestamp_ms: timestamp passed to detect_async for this frame for tracking
    """
    global latest_annotated_bgr, latest_timestamp
    latest_timestamp = timestamp_ms

    # Convert MediaPipe output image to an RGB NumPy view (zero-copy where possible)
    rgb_view = output_image.numpy_view()  # shape (H, W, 3), dtype=uint8, color order = RGB

    # Use your helper to draw the 21 landmarks + connections and handedness text
    annotated_rgb = draw_landmarks_on_image(rgb_view, result)

    # OpenCV expects BGR for display/write → convert once here
    latest_annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)


class MediaPipeCfg:
    """
    MediaPipe configuration for Hand Landmarker model
    """
    def __init__(self, model_path: str = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File {model_path} does not exist")

        self.ModelPath = model_path
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.NumHands = 2

    def create_options(self):
        HandLandmarkerOptions = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.ModelPath),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            num_hands=self.NumHands,
            result_callback=annotate_frame
        )
        return HandLandmarkerOptions


class CameraCfg:
    """
    Camera configuration to set up a livestream input using webcam
    Output is defined using VideoWriter
    """
    def __init__(self, cameraIdx: int, fps: float, is_array: bool):
        self.CameraIdx = cameraIdx # default camera (usually webcam)
        self.FPS = fps
        self.OutputPath = "output.mp4" if not is_array else "coordinates.csv"

    def create_camera(self):
        cam = cv2.VideoCapture(self.CameraIdx, cv2.CAP_DSHOW)
        if not cam.isOpened():
            raise RuntimeError("ERROR: could not open camera")
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") # define codec
        out = cv2.VideoWriter(self.OutputPath, fourcc, self.FPS, (frame_width, frame_height))
        return cam, out, frame_width, frame_height


def demo_landmarker(
    model_path: str = 'pretrained_model/hand_landmarker.task',
    cameraIdx: int = 1,
    fps: float = 20.0,
    is_array: bool = False
):
    MediaPipeCFG = MediaPipeCfg(model_path)
    CameraCFG = CameraCfg(cameraIdx=cameraIdx, fps=fps, is_array=is_array)

    cam, write_output, frame_width, frame_height = CameraCFG.create_camera()
    options = MediaPipeCFG.create_options()

    # Timestamp helper defined in function
    t0 = time.perf_counter()
    def now_ms():
        return int((time.perf_counter() - t0) * 1000)

    with MediaPipeCFG.HandLandmarker.create_from_options(options=options) as landmarker:
        while True:
            ok, frame_bgr = cam.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB (RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=frame_rgb)  # Wrap RGB frame to MP-readable format (MP Image)

            # Feed MP Image to landmarker asynchronously - requires timestamp
            landmarker.detect_async(mp_image, now_ms())

            display_frame = latest_annotated_bgr if latest_annotated_bgr is not None else frame_bgr

            if display_frame.shape[1] != frame_width or display_frame.shape[0] != frame_height:
                write_frame = cv2.resize(display_frame, (frame_width, frame_height))
            else:
                write_frame = display_frame

            cv2.imshow("MediaPipe Hands", display_frame)
            write_output.write(write_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cam.release()
    write_output.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        demo_landmarker()
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()