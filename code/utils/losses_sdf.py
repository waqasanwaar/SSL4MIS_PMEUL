# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 13:45:03 2025
"""
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

import torch
import torch.nn as nn

def compute_sdf(img_gt, out_shape):

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf

    return normalized_sdf

def compute_single_sdf(img):
    
    posmask = img.astype(np.bool)
    if posmask.any():
        negmask = ~posmask
        posdis = distance(posmask)
        negdis = distance(negmask)
        boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
        sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
        sdf[boundary==1] = 0
        normalized_sdf = sdf

        return normalized_sdf
    else:
        return posmask
    

def compute_mc_sdf(mask,out_shape,num_classes=4):
    sdf_all = np.zeros(out_shape)
    for b in range(out_shape[0]): # batch size
        for class_id in range(num_classes):
            slic=mask[b,].long().cpu().numpy()
            class_mask = (slic == class_id)#.astype(np.bool)
            sdf = compute_single_sdf(class_mask) 
            sdf_all[b,class_id,]=sdf
    return sdf_all
    

if __name__ == "__main__":
    import torch
    a = torch.ones(12, 4,256,256)
    b = torch.rand((12, 4,256,256), dtype=torch.float64) * 20.
    
    with torch.no_grad():
        gt_dis = compute_sdf(a.cpu().numpy(), b.shape)
        gt_dis = torch.from_numpy(gt_dis).float().cuda()
    #loss_sdf = mse_loss(outputs_tanh[:labeled_bs, 0, ...], gt_dis)
