import cv2
from detectron2.config import get_cfg
from detectron2.modeling import build_model, META_ARCH_REGISTRY
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import ImageList, Instances
from typing import Dict, List, Optional, Tuple

from detectron2.config import get_cfg
from detectron2.modeling import build_model, RPN_HEAD_REGISTRY
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.proposal_generator.rpn import StandardRPNHead

import os
import datetime
import numpy as np
from scipy.spatial import distance
import torch
from matplotlib import pyplot as plt

"""
This script is used to get the heatmap of the Faster R-CNN model in Detectron2
Got the idea form https://medium.com/@hirotoschwert/digging-into-detectron-2-part-4-3d1436f91266
"""

# This is the path to the models files, it should contain the config.yaml and model_final.pth
#MODEL_PATH = "./Fine_tuned_Detectron2/models/CA_models/50_final/"
MODEL_PATH = "./Fine_tuned_Detectron2/models/Glass_models/49_final/"
# This is the path to the images we want to analyze
FILES_PATH = "./Fine_tuned_Detectron2/data/Evaluation/detectadas/vidrio/"
# This is the path where the images will be saved
OUTPUT_FOLDER = "./Fine_tuned_Detectron2/data/Evaluation/detectadas/vidrio/"
# This is set to resize the images
#RESIZE = (1280, 720)
RESIZE = (1000, 750)

####################################################################
# This is the code that needs to be added to the detectron2 library
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
        #axis, fig = plt.subplots(2, 3, figsize=(20, 4))
        for i,(k, v) in enumerate(features.items()):
            v_ = v[:, 0].cpu().numpy()
            v_ = (v_ / 16 + 0.5) * 255
            v_ = np.asarray(v_.clip(0, 255), dtype=np.uint8).transpose((1, 2, 0))
            cv2.imwrite(OUTPUT_FOLDER +'heatmap_FPN/' + k + '.png', v_)
            plt.imshow(v_)
            plt.title("Feature map of {}".format(k))
            plt.axis('off')
            plt.savefig(OUTPUT_FOLDER + 'heatmap_FPN/heatmap_{}.png'.format(k))
            #fig[i%2][i%3].imshow(v_)
            #fig[i%2][i%3].set_title(k)
            #fig[i%2][i%3].axis('off')
        #plt.savefig(OUTPUT_FOLDER + 'heatmap_FPN/heatmap.png')
        #plt.close()

@RPN_HEAD_REGISTRY.register()
class MyRPNHead(StandardRPNHead):
    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        i=0
        for x in features:
            t = self.conv(x)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
            o = pred_objectness_logits[-1].sigmoid() * 255
            o = o.cpu().detach().numpy()[0, 1]
            o = cv2.resize(o, (320, 184))
            now = datetime.datetime.now()
            cv2.imwrite(OUTPUT_FOLDER + 'objectness/' + str(i)+'.png', np.asarray(o, dtype=np.uint8))
            i+=1
        return pred_objectness_logits, pred_anchor_deltas
####################################################################
    
# This function is used to load the model, as each part needs to load different modules, we load a model per image per part we want to analyze
# This is extremely inefficient and I know make no sense
def get_model(roi=False, fpn=False):
    cfg = get_cfg()
    cfg.merge_from_file(MODEL_PATH + "config.yaml")
    cfg.MODEL.WEIGHTS = MODEL_PATH + "model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    if roi:
        cfg.MODEL.RPN.HEAD_NAME = "MyRPNHead"
    if fpn:
        cfg.MODEL.META_ARCHITECTURE = "MyGeneralizedRCNN"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = build_model(cfg)
    model.to(DEVICE)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model

# This function is used to get the heatmap of the Region of Interest (ROI) of the image
# It saves a plot conatining the original image and the heatmap of each level of the pyramid
def get_heatmap_roi(file):
    model = get_model(roi=True)
    output_folder = os.path.join(OUTPUT_FOLDER, "heatmap_ROI")
    objectness_folder = os.path.join(OUTPUT_FOLDER, "objectness")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(objectness_folder, exist_ok=True)
    
    model.eval()
    imgs = []
    file_path = os.path.join(FILES_PATH, file + ".jpg")
    img = cv2.imread(file_path)  #BGR image
    
    img = cv2.resize(img, RESIZE)
    imgs.append(img)
    
    with torch.no_grad():
        inputs = {"image": torch.tensor(img).permute(2, 0, 1).float()}
        outputs = model([inputs])

    for i in range(5):  # five levels
        objectness = cv2.imread(os.path.join(objectness_folder + "/{}.png".format(i)))
        heatmap = cv2.applyColorMap(objectness, cv2.COLORMAP_JET)
        os.rename(objectness_folder + "/{}.png".format(i), objectness_folder + "/{}_p{}.png".format(file, i))
        imgs.append(cv2.resize(imgs[0], (320, 184)) // 2 + heatmap // 2)  # blending
    fig = plt.figure(figsize=(16, 7))
    for i, img in enumerate(imgs):
        fig.add_subplot(2, 3, i + 1)
        if i > 0:
            plt.imshow(img[0:-1, :, ::-1])  # ::-1 removes the edge
            plt.title("objectness on P" + str(i + 1))
        else:
            plt.imshow(img[:, :, ::-1])
            plt.title("input image")
    plt.savefig(os.path.join(output_folder,"{}.png".format(file)))
    plt.close()
    print("ROI heatmap of image {} saved in {}".format(file,output_folder))

# This function is used to get the heatmap of the Feature Pyramid Network (FPN) of the image
# It saves an image for each level of the pyramid. For now this does not show lots of information
def get_heatmap_fpn(file):
    model = get_model(fpn=True)
    output_folder = os.path.join(OUTPUT_FOLDER, "heatmap_FPN")
    os.makedirs(output_folder, exist_ok=True)
    model.eval()
    imgs = []
    file_path = os.path.join(FILES_PATH, file + ".jpg")
    img = cv2.imread(file_path)  #BGR image
    img = cv2.resize(img, RESIZE)
    imgs.append(img)
    
    with torch.no_grad():
        inputs = {"image": torch.tensor(img).permute(2, 0, 1).float()}
        outputs = model([inputs])
    
    for i in range(2,7):
        os.rename(output_folder + '/p{}.png'.format(i), output_folder + '/{}_p{}.png'.format(file, i))
        os.rename(output_folder + '/heatmap_p{}.png'.format(i), output_folder + '/heatmap_{}_p{}.png'.format(file,i))
    
    #os.rename(output_folder + '/heatmap.png', output_folder + '/{}.png'.format(file))
    print("FPN heatmap of image {} saved in {}".format(file,output_folder))

# This function is used to get for each prediction in the image the heatmap of the mask and the original image with the bounding box included
def get_heatmap_mask(file):
    model = get_model(MODEL_PATH)
    output_folder = os.path.join(OUTPUT_FOLDER, "heatmap_mask")
    os.makedirs(output_folder, exist_ok=True)
    
    model.eval()
    file_path = os.path.join(FILES_PATH, file + ".jpg")
    img = cv2.imread(file_path)  #BGR image
    img = cv2.resize(img, RESIZE)

    with torch.no_grad():
        inputs = {"image": torch.tensor(img).permute(2, 0, 1).float()}
        outputs = model.inference([inputs],do_postprocess=False)
    
    pred_masks = outputs[0].get_fields()['pred_masks'].cpu().numpy()

    if pred_masks.shape[0] == 0:
        return
    pred_boxes = outputs[0].get_fields()['pred_boxes'].tensor.cpu().numpy()
    
    pred_masks = pred_masks.squeeze(1)  # Resulting shape 28,28 in all, this is not the original shape of the mask
    
    #Show the original image to the left and the heatmap to the right
    for i in range(pred_masks.shape[0]):
        pred_mask = pred_masks[i].copy()
        x1,y1,x2,y2 = pred_boxes[i]
        x1,y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        orig_shape_pred_mask = cv2.resize(pred_mask, (x2-x1,y2-y1))  # Reshape the mask to the original shape, cv2 reescales the image to the new shape
        copy_img = img.copy()
        cv2.rectangle(copy_img, (x1,y1), (x2,y2), (0,255,0), 2)
        fig = plt.figure(figsize=(16, 7))
        fig.add_subplot(1, 2, 1)
        plt.imshow(copy_img)
        fig.add_subplot(1, 2, 2)
        plt.imshow(orig_shape_pred_mask , cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("Heatmap of image {}".format(file))
        plt.savefig(output_folder + "/{}_{}.png".format(file,i))
        plt.close()
    print("Mask heatmap of image {} saved in {}".format(file,output_folder))


def main():
    files = os.listdir(FILES_PATH)
    files = [file.split('.')[0] for file in files]
    #Make sure the folders exist
    os.makedirs(OUTPUT_FOLDER + "heatmap_mask", exist_ok=True)
    os.makedirs(OUTPUT_FOLDER + "heatmap_FPN", exist_ok=True)
    os.makedirs(OUTPUT_FOLDER + "heatmap_ROI", exist_ok=True)
    for file in files:
        get_heatmap_mask(file)
        get_heatmap_fpn(file)
        get_heatmap_roi(file)
    print("Done")

if __name__ == "__main__":
    main()