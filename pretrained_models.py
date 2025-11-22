import os
import torch
import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
from typing import Optional, Dict, Any, Union, List



class MediaPipeCfg:
    """
    MediaPipe configuration options for Hand Landmarker model

    To get model task, run:
    MediaPipeCFG = MediaPipeCfg(model_path)
    options = MediaPipeCFG.options
    HandLandmarker = MediaPipeCFG.HandLandmarker.create_from_options(options)
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
            running_mode=self.VisionRunningMode.IMAGE,
            num_hands=self.NumHands,
        )
        return HandLandmarkerOptions



ImageType = Union[str, "np.ndarray"]  # path or image
BBoxesType = Optional[List[Dict[str, Any]]]  # list of bounding boxes

class MMPoseCfg:
    """
    MMPose configuration options for initializing an OpenMMLab pose estimation model.
    Provides finer controls for general inference and environment options to adjust
    between local and cloud usage.

    You can create both the body and face pose estimators using MMPoseCfg.
    """
    def __init__(self, checkpoint_path: str = None,
                 config_path: str = None,
                 device: str = "cpu",
                 scope: str = "mmpose",
                 bbox_threshold: float = 0.5,
                 keypoint_threshold: float = 0.5,
                 nms_threshold: float = 0.5,
                 prediction_output_dir: str = None,
                 batch_size: int = 1,
                 cudnn_benchmark: bool = False,
                 np_start_method: str = "fork",
                 opencv_num_threads: int = 1,
                 dist_backend: str = "nccl"
                 ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device

        self.scope: str = scope
        self.bbox_threshold = bbox_threshold
        self.keypoint_threshold = keypoint_threshold
        self.nms_threshold = nms_threshold
        self.prediction_output_dir = prediction_output_dir
        self.batch_size = batch_size

        self.cudnn_benchmark = cudnn_benchmark
        self.np_start_method = np_start_method
        self.opencv_num_threads = opencv_num_threads
        self.dist_backend = dist_backend

    def file_check(self):
        # Check local file paths
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

    def create_model(self):
        assert self.file_check is True

        register_all_modules()

        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        cv2.setNumThreads(self.opencv_num_threads)

        model = init_model(self.config_path,
                           self.checkpoint_path,
                           device=self.device)
        return model

    def topdown_infer(
        self,
        model,
        img: ImageType,
        bboxes: BBoxesType = None,
        **kwargs
    ):
        """
        Run top-down pose inference on a single image/frame.

        Args:
            model: model returned by `create_model()`
            img: image path or numpy array image (RGB)
            bboxes: optional list of dicts with 'bbox' fields; if None,
                    the whole image is treated as a single bbox.
            **kwargs: forwarded to `inference_topdown`

        Returns:
            List[PoseDataSample]: Ground truth keypoint annotations,
                                  predictions,
                                  heatmap/PAF annotations.
        """
        data_samples = inference_topdown(
            model=model,
            img=img,
            bboxes=bboxes,
            **kwargs
        )
        return data_samples