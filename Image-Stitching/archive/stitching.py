'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image


'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image. 
    """
    """
    Please do not use cv2, numpy, PIL, imageio for any image related operations. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    
    keys = sorted(imgs.keys())
    img1 = imgs[keys[0]].float()/255.0
    img2 = imgs[keys[1]].float()/255.0

    gray1 = K.color.rgb_to_grayscale(img1.unsqueeze(0))
    gray2 = K.color.rgb_to_grayscale(img2.unsqueeze(0))
    
    #feature matching
    sift = K.feature.SIFTFeature(num_features=800, rootsift=True)

    lafs1, resps1, descs1 = sift.forward(gray1)  #lafs: B,N,2,3 AND desc: B, N, D
    lafs2, resps2, descs2 = sift.forward(gray2)
    
    print(descs1.shape, descs2.shape)
    
    desc1_ = descs1[0]
    desc2_ = descs2[0]  
    
    lafs1_=lafs1[0]
    lafs2_=lafs2[0]
    
    matcher = K.feature.DescriptorMatcher("smnn", 0.8)
    dists, match_idx = matcher.forward(desc1_, desc2_) #matches: N,2
    
    if match_idx.shape[0] == 0:
        return ((img1 + img2)/2 *255).to(torch.uint8).cpu()
    
    idx1 = match_idx[:,0]
    idx2 = match_idx[:,1]
    
    matched_lafs1 = lafs1_[idx1]
    matched_lafs2 = lafs2_[idx2]
    
    print("matched lafs1 shape", matched_lafs1.shape)
    
    laf_center1 = K.feature.get_laf_center(matched_lafs1.unsqueeze(0))
    laf_center_2 = K.feature.get_laf_center(matched_lafs2.unsqueeze(0))

    print("laf center1 shape", laf_center1.shape)
    
    # overlapped_1 = lafs1[0, idx1, :, 2]
    # overlapped_2 = lafs2[0, idx2, :, 2]
    
    #ransac
    ransac = K.geometry.ransac.RANSAC(model_type='homography')
    model, inliers = ransac.forward(kp1=laf_center_2.squeeze(0), kp2 = laf_center1.squeeze(0))
    print("model shape", model.shape)
    print("inliers: ",int(inliers.sum()))
    
    print("shape is", img1.shape, img2.shape)
    
    h1, w1 = img1.shape[-2:]
    h2, w2 = img2.shape[-2:]
    
    H_identity = torch.eye(3, dtype=torch.float32)
    corners1 = corners(h1,w1,H_identity)
    corners2 = corners(h2, w2, model)
    all_corners = torch.cat([corners1, corners2], dim=1)
    
    xmin, xmax = all_corners[0].min(), all_corners[0].max()
    ymin, ymax = all_corners[1].min(), all_corners[1].max() 
    
    canvas_w = int(torch.ceil(xmax - xmin).item())
    canvas_h = int(torch.ceil(ymax - ymin).item())
    
    translation = torch.tensor([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=torch.float32)
   
    H_final1=translation@H_identity
    H_final2=translation@model
    
    warp1 = K.geometry.transform.warp_perspective(img1.unsqueeze(0), H_final1.unsqueeze(0), dsize=(canvas_h, canvas_w))[0]
    print(warp1.shape)
    warp2 = K.geometry.transform.warp_perspective(img2.unsqueeze(0), H_final2.unsqueeze(0), dsize=(canvas_h, canvas_w))[0]
    print(warp2.shape)
    mask1 = (warp1.sum(0, keepdim=True) > 0).float()
    mask2 = (warp2.sum(0, keepdim=True) > 0).float()
    
    canvas = torch.zeros((3, canvas_h, canvas_w), dtype=torch.float32)
    total_mask = torch.zeros((3, canvas_h, canvas_w), dtype=torch.float32)
    canvas+= warp1*mask1
    canvas+= warp2*mask2
    
    mask3 = torch.cat([mask1, mask1, mask1], dim=0)
    mask4 = torch.cat([mask2, mask2, mask2], dim=0)
    total_mask += mask3
    total_mask += mask4
    total_mask = total_mask.clamp(min=1.0)
    stitched = canvas/total_mask
    
    return (stitched * 255).to(torch.uint8).cpu()



def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    
    keys = sorted(imgs.keys())
    num_img = len(keys)
    # img1 = imgs[keys[0]].float()/255.0
    # img2 = imgs[keys[1]].float()/255.0
    # img3 = imgs[keys[2]].float()/255.0
    # img4 = imgs[keys[3]].float()/255.0
    images=[]
    for k in keys:
        img = imgs[k].float()/255.0
        images.append(img)
        
    # gray1 = K.color.rgb_to_grayscale(img1.unsqueeze(0))
    # gray2 = K.color.rgb_to_grayscale(img2.unsqueeze(0))
    # gray3 = K.color.rgb_to_grayscale(img3.unsqueeze(0))
    # gray4 = K.color.rgb_to_grayscale(img4.unsqueeze(0))
    
    sift = K.feature.SIFTFeature(num_features=800, rootsift=True)
    
    # lafs1, resp1, desc1 = sift(gray1)
    # lafs2, resp2, desc2 = sift(gray2)
    # lafs3, resp3, desc3 = sift(gray3)
    # lafs4, resp4, desc4 = sift(gray4)
    
    lafs = []
    descs = []
 
    for i in images:
        grayscale = K.color.rgb_to_grayscale(i.unsqueeze(0))
        laf, resp, desc = sift(grayscale)
        lafs.append(laf[0])
        descs.append(desc[0])
    
    # lafs1_ = lafs1[0]
    # lafs2_ = lafs2[0]
    # lafs3_ = lafs3[0]   
    # lafs4_ = lafs4[0]
    
    # desc1_ = desc1[0]
    # desc2_ = desc2[0]
    # desc3_ = desc3[0]
    # desc4_ = desc4[0]
    
    # #1&2
    # H12, inliers12 = matching(desc1_, desc2_, lafs1_, lafs2_ )
    
    # #2&3
    # H23, inliers23 = matching(desc2_, desc3_, lafs2_, lafs3_)
    
    # #3&4
    # H34, inliers34 = matching(desc3_, desc4_, lafs3_, lafs4_)
    
    # homo = []
    # for i in range(num_img-1):
    #     H, inliers = matching(descs[i], descs[i+1], lafs[i], lafs[i+1])
    #     homo.append(H)
        
    H_pair={}
    for i in range(num_img):
        for j in range(num_img):
            if i==j:
                H_pair[(i,j)]= torch.eye(3)
            elif j>i:
                H, inliers = matching(descs[i], descs[j], lafs[i], lafs[j])
                H_pair[(i,j)] = H
                H_pair[(j,i)] = torch.linalg.inv(H)
    
    # Use image 0 as reference, get direct transformations
    reference_idx=0
    H_identity = []
    for i in range(num_img):
        if i==reference_idx:
            H_identity.append(torch.eye(3, dtype=torch.float32))
        else:
            H_identity.append(H_pair[(reference_idx, i)])
    
    all_corners = []
    for i, img in enumerate(images):
        h, w = img.shape[-2:]
        corner = corners(h, w, H_identity[i])
        all_corners.append(corner)
    
    # print(f"H_a shape: {H_a.shape}")
    # print(f"H_b shape: {H_b.shape}")
    # print(f"H_c shape: {H_c.shape}")
        
    # h1, w1 = img1.shape[-2:]
    # h2, w2 = img2.shape[-2:]
    # h3, w3 = img3.shape[-2:]
    # h4, w4 = img4.shape[-2:]
    
    # corners1 = corners(h1, w1, H_identity)
    # corners2 = corners(h2, w2, H_a)
    # corners3 = corners(h3, w3, H_b)
    # corners4 = corners(h4, w4, H_c)
    
    #all_corners = torch.vstack([corners1, corners2, corners3, corners4])
    # all_corners = torch.cat([corners1, corners2, corners3, corners4], dim=1)
    all_corners = torch.cat(all_corners, dim=1)
    xmin, xmax = all_corners[0].min(), all_corners[0].max()
    ymin, ymax = all_corners[1].min(), all_corners[1].max()
    
    final_w = int(torch.ceil(xmax - xmin).item())
    final_h = int(torch.ceil(ymax - ymin).item())
    
    #refer slide29 for translation
    
    translation = torch.tensor([[1, 0, -xmin], [0, 1, -ymin], [0,0,1]], dtype=torch.float32)
    
    # H_final1=translation@H_identity
    # H_final2=translation@H_a    
    # H_final3=translation@H_b
    # H_final4=translation@H_c
    
    all_warps=[]
    all_masks=[]
    for i, img in enumerate(images):
        H_translated = translation@H_identity[i]
        warp = K.geometry.transform.warp_perspective(img.unsqueeze(0), H_translated.unsqueeze(0), (final_h, final_w)).squeeze(0)
        mask = (warp.sum(0, keepdim=True)>0).float()
        all_warps.append(warp)
        all_masks.append(mask)
    
    # warp1 = K.geometry.transform.warp_perspective(img1.unsqueeze(0), H_final1.unsqueeze(0), (final_h, final_w)).squeeze(0)
    # warp2 = K.geometry.transform.warp_perspective(img2.unsqueeze(0), H_final2.unsqueeze(0), (final_h, final_w)).squeeze(0)
    # warp3 = K.geometry.transform.warp_perspective(img3.unsqueeze(0), H_final3.unsqueeze(0), (final_h, final_w)).squeeze(0)
    # warp4 = K.geometry.transform.warp_perspective(img4.unsqueeze(0), H_final4.unsqueeze(0), (final_h, final_w)).squeeze(0)
    
    print(all_warps[0].shape)

    pano = torch.zeros((3, final_h, final_w), dtype=torch.float32)
    total_mask = torch.zeros((3, final_h, final_w), dtype=torch.float32)
    
    for warp, mask in zip(all_warps, all_masks):
        mask_repeat = torch.cat([mask, mask, mask],dim=0)
        pano+=warp*mask_repeat
        total_mask+=mask_repeat
    
    # mask1 = (warp1.sum(0, keepdim=True) > 0).float()
    # mask2 = (warp2.sum(0, keepdim=True) > 0).float()
    # mask3 = (warp3.sum(0, keepdim=True) > 0).float()
    # mask4 = (warp4.sum(0, keepdim=True) > 0).float()
    
    # print(f"mask1 sum: {mask1.sum()}")
    # print(f"mask2 sum: {mask2.sum()}")
    # print(f"mask3 sum: {mask3.sum()}")
    # print(f"mask4 sum: {mask4.sum()}")

    # mask_fin1= torch.cat([mask1, mask1, mask1], dim=0)
    # mask_fin2= torch.cat([mask2, mask2, mask2], dim=0)
    # mask_fin3= torch.cat([mask3, mask3, mask3], dim=0)
    # mask_fin4= torch.cat([mask4, mask4, mask4], dim=0)
    
    # total_mask = torch.zeros((3, final_h, final_w), dtype=torch.float32)
    
    # pano+=warp1*mask_fin1
    # pano+=warp2*mask_fin2
    # pano+=warp3*mask_fin3
    # pano+=warp4*mask_fin4
    
    # total_mask+=mask_fin1
    # total_mask+=mask_fin2
    # total_mask+=mask_fin3
    # total_mask+=mask_fin4
    for i in range(3):        # R, G, B
        for j in range(final_h):
            for k in range(final_w):
                if total_mask[i,j,k] == 0:
                    total_mask[i,j,k] = 1

                                
    panoroma = pano/total_mask
    
    for i in range(3):
        for j in range(final_h):
            for k in range(final_w):
                if panoroma[i, j,k]<0:
                    panoroma[i, j,k]=0
                elif panoroma[i, j,k]>1:
                    panoroma[i, j,k]=1
    
    # Compute overlap matrix
    overlap_matrix = torch.zeros((num_img, num_img), dtype=torch.float32)
    for i in range(num_img):
        area_i = all_masks[i].sum()
        for j in range(num_img):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                overlap_area = (all_masks[i] * all_masks[j]).sum()
                if area_i > 0:
                    overlap_matrix[i, j] = (overlap_area / area_i).item()


             
    panoroma = (panoroma * 255).byte().cpu()
    return panoroma, overlap_matrix

def matching(descA, descB, lafs1_, lafs2_):
    matcher = K.feature.DescriptorMatcher("smnn", 0.8)
    dists, match_idx = matcher.forward(descA, descB)
    if match_idx.shape[0] <4:
        return torch.eye(3, dtype=torch.float32), torch.tensor([0])
    
    idx1 = match_idx[:,0]
    idx2 = match_idx[:,1]
    
    matched_lafs1 = lafs1_[idx1]
    matched_lafs2 = lafs2_[idx2]
    
    print("matched lafs1 shape", matched_lafs1.shape)
    
    laf_center_1 = K.feature.get_laf_center(matched_lafs1.unsqueeze(0))
    laf_center_2 = K.feature.get_laf_center(matched_lafs2.unsqueeze(0))

    print("laf center1 shape", laf_center_1.shape)
    
    # overlapped_1 = lafs1[0, idx1, :, 2]
    # overlapped_2 = lafs2[0, idx2, :, 2]
    
    #ransac
    ransac = K.geometry.ransac.RANSAC(model_type='homography')
    H, inliers = ransac.forward(kp1=laf_center_1.squeeze(0), kp2 = laf_center_2.squeeze(0))
    
    
    if H is None or inliers.sum() < 4:
        return torch.eye(3, dtype=torch.float32), inliers
        
    return H.squeeze(0), inliers

def corners(h2, w2, H):
    # corners1 = torch.tensor([[0, h1-1], [w1-1, h1-1], [0,0], [w1-1, 0]]).float()
    corners2 = torch.tensor([[0, h2-1, 1], [w2-1, h2-1, 1], [0,0, 1], [w2-1, 0, 1]], dtype=torch.float32).T
    transformed = H @ corners2
    trans_corners = transformed[:2]/transformed[2:3]
    
    return trans_corners
    
    
    
