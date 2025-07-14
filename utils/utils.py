import warnings
import numpy as np
import math
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image


def unnormalize_coords(x_n,h,w):
    x = torch.stack(
        (w * (x_n[..., 0] + 1) / 2, h * (x_n[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    return x

def get_tuple_transform_ops(resize=None, mode=InterpolationMode.BICUBIC, normalize=True, unscale=False, clahe = False, colorjiggle_params = None):
    ops = []
    if resize:
        ops.append(TupleResize(resize, mode=mode))
    ops.append(TupleToTensorScaled())
    if normalize:
        ops.append(
            TupleNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )  # Imagenet mean/std
    return TupleCompose(ops)

class ToTensorScaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]"""

    def __call__(self, im):
        if not isinstance(im, torch.Tensor):
            im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
            im /= 255.0
            return torch.from_numpy(im)
        else:
            return im

    def __repr__(self):
        return "ToTensorScaled(./255)"


class TupleToTensorScaled(object):
    def __init__(self):
        self.to_tensor = ToTensorScaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorScaled(./255)"


class ToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __call__(self, im):
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self):
        return "ToTensorUnscaled()"


class TupleToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __init__(self):
        self.to_tensor = ToTensorUnscaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorUnscaled()"

class TupleResizeNearestExact:
    def __init__(self, size):
        self.size = size
    def __call__(self, im_tuple):
        return [F.interpolate(im, size = self.size, mode = 'nearest-exact') for im in im_tuple]

    def __repr__(self):
        return "TupleResizeNearestExact(size={})".format(self.size)


class TupleResize(object):
    def __init__(self, size, mode=InterpolationMode.BICUBIC):
        self.size = size
        self.resize = transforms.Resize(size, mode, antialias=None)
    def __call__(self, im_tuple):
        return [self.resize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleResize(size={})".format(self.size)
    
class Normalize:
    def __call__(self,im):
        mean = im.mean(dim=(1,2), keepdims=True)
        std = im.std(dim=(1,2), keepdims=True)
        return (im-mean)/std


class TupleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im_tuple):
        c,h,w = im_tuple[0].shape
        if c > 3:
            warnings.warn(f"Number of channels c={c} > 3, assuming first 3 are rgb")
        return [self.normalize(im[:3]) for im in im_tuple]

    def __repr__(self):
        return "TupleNormalize(mean={}, std={})".format(self.mean, self.std)


class TupleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_tuple):
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

@torch.no_grad()
def cls_to_flow(cls, deterministic_sampling = True):
    B,C,H,W = cls.shape
    device = cls.device
    res = round(math.sqrt(C))
    G = torch.meshgrid(
        *[torch.linspace(-1+1/res, 1-1/res, steps = res, device = device) for _ in range(2)],
        indexing = 'ij'
        )
    G = torch.stack([G[1],G[0]],dim=-1).reshape(C,2)
    if deterministic_sampling:
        sampled_cls = cls.max(dim=1).indices
    else:
        sampled_cls = torch.multinomial(cls.permute(0,2,3,1).reshape(B*H*W,C).softmax(dim=-1), 1).reshape(B,H,W)
    flow = G[sampled_cls]
    return flow

@torch.no_grad()
def cls_to_flow_refine(cls):
    B,C,H,W = cls.shape
    device = cls.device
    res = round(math.sqrt(C))
    G = torch.meshgrid(
        *[torch.linspace(-1+1/res, 1-1/res, steps = res, device = device) for _ in range(2)],
        indexing = 'ij'
        )
    G = torch.stack([G[1],G[0]],dim=-1).reshape(C,2)
    # FIXME: below softmax line causes mps to bug, don't know why.
    if device.type == 'mps':
        cls = cls.log_softmax(dim=1).exp()
    else:
        cls = cls.softmax(dim=1)
    mode = cls.max(dim=1).indices
    
    index = torch.stack((mode-1, mode, mode+1, mode - res, mode + res), dim = 1).clamp(0,C - 1).long()
    neighbours = torch.gather(cls, dim = 1, index = index)[...,None]
    flow = neighbours[:,0] * G[index[:,0]] + neighbours[:,1] * G[index[:,1]] + neighbours[:,2] * G[index[:,2]] + neighbours[:,3] * G[index[:,3]] + neighbours[:,4] * G[index[:,4]]
    tot_prob = neighbours.sum(dim=1)  
    flow = flow / tot_prob
    return flow

imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])


def numpy_to_pil(x: np.ndarray):
    """
    Args:
        x: Assumed to be of shape (h,w,c)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.max() <= 1.01:
        x *= 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def tensor_to_pil(x, unnormalize=False):
    if unnormalize:
        x = x * (imagenet_std[:, None, None].to(x.device)) + (imagenet_mean[:, None, None].to(x.device))
    x = x.detach().permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cpu()
    return batch


def get_pose(calib):
    w, h = np.array(calib["imsize"])[0]
    return np.array(calib["K"]), np.array(calib["R"]), np.array(calib["T"]).T, h, w


def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ (R1.T)
    trans = -rots @ t1 + t2
    return rots, trans

@torch.no_grad()
def reset_opt(opt):
    for group in opt.param_groups:
        for p in group['params']:
            if p.requires_grad:
                state = opt.state[p]
                # State initialization

                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p)
                # Exponential moving average of gradient difference
                state['exp_avg_diff'] = torch.zeros_like(p)


def flow_to_pixel_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                w1 * (flow[..., 0] + 1) / 2,
                h1 * (flow[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    return flow

to_pixel_coords = flow_to_pixel_coords # just an alias

def flow_to_normalized_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                2 * (flow[..., 0]) / w1 - 1,
                2 * (flow[..., 1]) / h1 - 1,
            ),
            axis=-1,
        )
    )
    return flow

to_normalized_coords = flow_to_normalized_coords # just an alias

def warp_to_pixel_coords(warp, h1, w1, h2, w2):
    warp1 = warp[..., :2]
    warp1 = (
        torch.stack(
            (
                w1 * (warp1[..., 0] + 1) / 2,
                h1 * (warp1[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    warp2 = warp[..., 2:]
    warp2 = (
        torch.stack(
            (
                w2 * (warp2[..., 0] + 1) / 2,
                h2 * (warp2[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    return torch.cat((warp1,warp2), dim=-1)

def get_grid(b, h, w, device):
    grid = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device)
            for n in (b, h, w)
        ],
        indexing = 'ij'
    )
    grid = torch.stack((grid[2], grid[1]), dim=-1).reshape(b, h, w, 2)
    return grid


def get_autocast_params(device=None, enabled=False, dtype=None):
    if device is None:
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        #strip :X from device
        autocast_device = str(device).split(":")[0]
    if 'cuda' in str(device):
        out_dtype = dtype
        enabled = True
    else:
        out_dtype = torch.bfloat16
        enabled = False
        # mps is not supported
        autocast_device = "cpu"
    return autocast_device, enabled, out_dtype



def register_grad_debug_hook_ddp(model):
    """
    给 DDP 模型注册 grad hook，自动在 backward 时检查 stride 和是否 contiguous。
    仅在 rank 0 打印信息。
    """
    import torch.distributed as dist
    rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    if not rank_zero:
        return

    print("[✅ Hook Registered] Monitoring gradient layout issues...")

    def make_hook(name):
        def hook_fn(grad):
            if not grad.is_contiguous():
                print(f"[⚠️ NON-CONTIGUOUS GRAD] {name}: shape={grad.shape}, stride={grad.stride()}")
            if hasattr(grad, '_param_stride'):
                if grad.stride() != grad._param_stride:
                    print(f"[⚠️ STRIDE MISMATCH] {name}: param.stride={grad._param_stride}, grad.stride={grad.stride()}")
        return hook_fn

    for name, param in model.module.named_parameters():
        if param.requires_grad:
            hook = make_hook(name)
            # 把 param 的原始 stride 存起来用于比较
            param.register_hook(hook)
            if param.grad is not None:
                param.grad._param_stride = param.stride()