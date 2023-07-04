from pathlib import Path
import numpy as np
import ultralytics.nn.tasks
import ultralytics.nn.autobackend
import torch

class UltralyticsInferer:
    def __init__(self, model_path: Path):
        # weights = ultralytics.nn.tasks.torch_safe_load(model_path)
        weights_with_extra_info, _ = ultralytics.nn.tasks.attempt_load_one_weight(model_path)
        #variables explaining not so better named parameters
        use_opencv_dnn = False
        yaml_path_with_class_names = None

        self._inferencing_model = ultralytics.nn.autobackend.AutoBackend(
            weights_with_extra_info,
            device="cpu",
            dnn=use_opencv_dnn,
            data=yaml_path_with_class_names,
            fp16=False,
            fuse=True,
            verbose=False)
        
        self._inferencing_model.eval()

    def detect(self, infer_data: np.array):
        im = torch.from_numpy(infer_data)
        # img = im.to(self.device)
        # img = img.half() if self.model.fp16 else img.float()
        all_results = self._inferencing_model(im)
        gathered_results = all_results[0].detach().cpu().numpy()
        return gathered_results
