from torchvision import models
import numpy as np
import cv2
import pdb
from PIL import Image
from torch.nn import functional as F
from torchvision import models, transforms
import io

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


# usage
if __name__ == '__main__':
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'

    net.eval()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # evaluate image
    img_pil = Image.open('utils/test.jpg')
    img_tensor = preprocess(img_pil)
    img_variable = img_tensor.unsqueeze(0)
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    img = cv2.imread('utils/test.jpg')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('CAM.jpg', result)
