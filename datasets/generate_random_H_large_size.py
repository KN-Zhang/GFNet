import torch
import kornia
import kornia.geometry.transform as KGT
from torchvision import transforms

def random_four_points(deform_area, w, h, img, bi=False, H=None):
    topleft = [
        torch.randint(0, deform_area, size=(1, )),
        torch.randint(0, deform_area, size=(1, ))
    ]
    topright = [
        torch.randint(w-deform_area, w, size=(1, )),
        torch.randint(0, deform_area, size=(1, ))
    ]
    botright = [
        torch.randint(w-deform_area, w, size=(1, )),
        torch.randint(h-deform_area, h, size=(1, ))
    ]
    botleft = [
        torch.randint(0, deform_area, size=(1, )),
        torch.randint(h-deform_area, h, size=(1, ))
    ]
    tgt_points = torch.tensor([[deform_area//2, deform_area//2], [w-deform_area//2-1, deform_area//2], [w-deform_area//2-1, h-deform_area//2-1], [deform_area//2, h-deform_area//2-1]]).float() ## 4x2
    
    if bi:
        src_points = torch.tensor([topleft, topright, botright, botleft]).float() ## 4x2
    else:
        src_points = tgt_points
    if H is None:
        H = KGT.get_perspective_transform(src_points.unsqueeze(0), tgt_points.unsqueeze(0)).squeeze(0) ## 3x3
    # flow_points = src_points - tgt_points
    
    warped_img = KGT.warp_perspective(img.unsqueeze(0), H.unsqueeze(0), (h, w)).squeeze(0) # C H W
    warped_img = warped_img[:, deform_area//2:h-deform_area//2, deform_area//2:w-deform_area//2] ## C 128 128
    
    return H, warped_img

def randomH(img1, img2, crop_size, input_size, deformation_ratio=0.33, bi=True):
    c1, h1, w1 = img1.shape
    c2, h2, w2 = img2.shape
    assert c1 == c2
    assert h1 == h2
    assert w1 == w2
    
    if w1<=crop_size or h1<=crop_size:
        size = crop_size + 10
        o_resize = transforms.Resize(size=size, interpolation=3, antialias=None) ##3 means bicubic
        img1, img2 = o_resize(img1), o_resize(img2)
    c1, h1, w1 = img1.shape    
    crop_top_left = [torch.randint(0, w1-crop_size, size=(1,)),
                     torch.randint(0, h1-crop_size, size=(1,))
                     ]
    img1 = img1[:, crop_top_left[1]:crop_top_left[1]+crop_size, crop_top_left[0]:crop_top_left[0]+crop_size]
    img2 = img2[:, crop_top_left[1]:crop_top_left[1]+crop_size, crop_top_left[0]:crop_top_left[0]+crop_size]
    
    c, h_original, w_original = img1.shape
    deform_area = int(w_original * deformation_ratio)
    
    H_1t, img1 = random_four_points(deform_area, w_original, h_original, img1, bi=True)
    H_2t, img2 = random_four_points(deform_area, w_original, h_original, img2, bi=bi)
        
    H_1t2t = H_2t @ H_1t.inverse()

    src_points = torch.tensor([[deform_area//2, deform_area//2], [w_original-deform_area//2-1, deform_area//2], [w_original-deform_area//2-1, h_original-deform_area//2-1], [deform_area//2, h_original-deform_area//2-1]]).float() ## 4x2
    tgt_points = kornia.geometry.transform_points(H_1t2t.unsqueeze(0), src_points.unsqueeze(0)).squeeze(0)
    flow = tgt_points - src_points
    _, h, w = img1.shape
    src_points = torch.tensor([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).float()
    tgt_points = src_points + flow
    
    H_s2t = KGT.get_perspective_transform(src_points.unsqueeze(0), tgt_points.unsqueeze(0)).squeeze(0) ## 3x3
    
    if input_size.size[0] != h or input_size.size[1] != w:
        img1, img2 = input_size(img1), input_size(img2)
        _, h_input, w_input = img1.shape

        H_s2t = torch.diag(torch.tensor([h_input/h, h_input/h, 1.])).float() @ \
        H_s2t @ \
        torch.diag(torch.tensor([w_input/w, w_input/w, 1.])).float().inverse()
    else:
        _, h_input, w_input = img1.shape
    
    warped_src_return = KGT.warp_perspective(img1.unsqueeze(0), H_s2t.unsqueeze(0), (h_input, w_input)).squeeze(0) # C H W
    
    return img2, img1, H_s2t, warped_src_return ## img1: src，img2: tgt


def crop(img1, img2, crop_size=512):
    c1, h1, w1 = img1.shape
    c2, h2, w2 = img2.shape
    assert c1 == c2
    assert h1 == h2
    assert w1 == w2
    
    if w1<=crop_size or h1<=crop_size:
        size = min(w1, h1) + 3
        o_resize = transforms.Resize(size=size, antialias=None)
        img1, img2 = o_resize(img1), o_resize(img2)
    c1, h1, w1 = img1.shape    
    crop_top_left = [torch.randint(0, w1-crop_size, size=(1,)),
                     torch.randint(0, h1-crop_size, size=(1,))
                     ]
    img1 = img1[:, crop_top_left[1]:crop_top_left[1]+crop_size, crop_top_left[0]:crop_top_left[0]+crop_size]
    img2 = img2[:, crop_top_left[1]:crop_top_left[1]+crop_size, crop_top_left[0]:crop_top_left[0]+crop_size]
    
    return img1, img2  
    

    