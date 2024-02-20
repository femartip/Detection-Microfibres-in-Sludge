import cv2
from detectron2.config import get_cfg
from detectron2.modeling import build_model, META_ARCH_REGISTRY
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import ImageList, Instances
from typing import Dict, List, Optional, Tuple

import os
import datetime
import numpy as np
from scipy.spatial import distance
import torch
from matplotlib import pyplot as plt

"""
Apply trained model on inference images to get objectness of rpn detectrions
For this, need to change file ./detectron2/modeling/proposal_generator/rpn.py
This saves the probability map of object existance for each level of the FPN
Got the ide form https://medium.com/@hirotoschwert/digging-into-detectron-2-part-4-3d1436f91266
"""
@META_ARCH_REGISTRY.register()
class MyGeneralizedRCNN(GeneralizedRCNN):
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        
        self.save_features_as_images(features)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
        
    @staticmethod
    def save_features_as_images(features):
        import numpy as np
        import cv2
        for k, v in features.items():
            v_ = v[:, 0].cpu().numpy()
            v_ = (v_ / 16 + 0.5) * 255
            v_ = np.asarray(v_.clip(0, 255), dtype=np.uint8).transpose((1, 2, 0))
            cv2.imwrite('./Fine_tuned_Detectron2/data/heatmap_FPN/' + k + '.png', v_)


model_path = "49_final"

cfg = get_cfg()
cfg.merge_from_file("./Fine_tuned_Detectron2/models/{}/config.yaml".format(model_path))
cfg.MODEL.WEIGHTS = os.path.join("./Fine_tuned_Detectron2/models/{}/model_final.pth".format(model_path))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.META_ARCHITECTURE = "MyGeneralizedRCNN"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model(cfg)
model.to(DEVICE)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

files = os.listdir('./Fine_tuned_Detectron2/data/inference')
files = [file.split('.')[0] for file in files]

for j,file in enumerate(files):
    
    model.eval()
    imgs = []
    img = cv2.imread("./Fine_tuned_Detectron2/data/inference/{}.jpg".format(file))  #BGR image
    img = cv2.resize(img, (1000, 750))
    imgs.append(img)
    
    with torch.no_grad():
        inputs = {"image": torch.tensor(img).permute(2, 0, 1).float()}
        outputs = model([inputs])
    
    for i in range(2,7):
        os.rename('./Fine_tuned_Detectron2/data/heatmap_FPN/p{}.png'.format(i), './Fine_tuned_Detectron2/data/heatmap_FPN/{}_p{}.png'.format(file, i))

print("Done")