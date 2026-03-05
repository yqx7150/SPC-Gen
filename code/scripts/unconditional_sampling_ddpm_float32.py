import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.models.diffusion.ddpm import DDPM
from ldm.util import instantiate_from_config

rescale = lambda x: (x + 1.) / 2.

# def custom_to_pil(x):
#     x = x.detach().cpu()
#     x = torch.clamp(x, -1., 1.)
#     x = (x + 1.) / 2.
#     x = x.permute(1, 2, 0).numpy()
#     x = (255 * x).astype(np.uint8)
#     x = Image.fromarray(x)
#     if not x.mode == "RGB":
#         x = x.convert("RGB")
#     return x
def custom_to_pil(x):
    # 保留原始浮点数数据（用于存储）
    # 可视化时临时归一化到[0, 255]并转为uint8
    x_np = x.detach().cpu().numpy()  # 保留float32原始数据
    # 归一化到[0, 255]用于显示（不改变原始数据）
    x_min, x_max = x_np.min(), x_np.max()
    x_norm = (x_np - x_min) / (x_max - x_min + 1e-8)  # 避免除零
    x_norm = (255 * x_norm).astype(np.uint8)
    # 处理通道（PET通常是单通道）
    if x_norm.ndim == 3:
        x_norm = x_norm.squeeze(0)  # 去除通道维度 (1, H, W) -> (H, W)
    img = Image.fromarray(x_norm, mode="L")  # 单通道灰度图
    return img, x_np  # 返回可视化图像和原始float32数据

# 单通道
# def custom_to_pil(x):
#     x = x.detach().cpu()  # 将张量从 GPU 移动到 CPU，并解除梯度跟踪
#     x = torch.clamp(x, -1., 1.)  # 将值限制在 [-1, 1] 范围内
#     x = (x + 1.) / 2.  # 将值映射到 [0, 1] 范围内
#     x = x.squeeze(0).numpy()  # 去掉通道维度 (1, H, W) -> (H, W)
#     x = (255 * x).astype(np.uint8)  # 将值缩放到 [0, 255] 并转换为 uint8 类型
#     x = Image.fromarray(x, mode="RGB")  # 创建一个灰度模式的 PIL 图像
#     return x
# def custom_to_np(x):
#     # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
#     sample = x.detach().cpu()
#     sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
#     sample = sample.permute(0, 2, 3, 1)
#     sample = sample.contiguous()
#     return sample
def custom_to_np(x):
    # 保留float32精度，不做[0,255]映射
    sample = x.detach().cpu()
    # 若模型输出是标准化的（如[-1,1]），需还原为PET原始范围（根据训练时的预处理逻辑）
    # 例如：假设训练时用 (x - mean) / std 标准化，此处需还原为 x = sample * std + mean
    # sample = sample * std + mean  # 关键：根据预处理反向操作还原真实值
    sample = sample.permute(0, 2, 3, 1).contiguous()  # 调整维度顺序
    return sample.numpy().astype(np.float32)  # 保存为float32


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=True, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape, eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=True, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir, '*.png')))-1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model,
                                             batch_size=batch_size,
                                             vanilla=vanilla,
                                             custom_steps=custom_steps,
                                             eta=eta)
            #n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample", np_path=nplog)
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        print(all_img.shape)
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional pre_trained_models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


# def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
#     for k in logs:
#         if k == key:
#             batch = logs[key]
#             if np_path is None:
#                 for x in batch:
#                     img = custom_to_pil(x)
#                     imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
#                     img.save(imgpath)
#                     n_saved += 1
#             else:
#                 npbatch = custom_to_np(batch)
#                 shape_str = "x".join([str(x) for x in npbatch.shape])
#                 nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
#                 np.savez(nppath, npbatch)
#                 n_saved += npbatch.shape[0]
#     return n_saved
# 在save_logs函数中，替换图像保存为numpy数组保存
def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            # 保存为float32的numpy数组（或转换为DICOM）
            npbatch = custom_to_np(batch)
            nppath = os.path.join(np_path, f"pet_sample_{n_saved:06}.npy")
            np.save(nppath, npbatch)
            n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
        

    )

    parser.add_argument(
        "-b",
        "--base",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )

    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=True,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="outputs"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"])
    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = opt.resume
    print(f'ckpt is {ckpt}')


    base_configs = [opt.base]
    #
    print("base_configs ", base_configs)

    config = OmegaConf.load(opt.base)

    locallog = opt.base.split(os.sep)[-2]  # unconditional

    print(f"locallog is {locallog}")
    logdir = os.path.join(opt.logdir, locallog)

    print(f'config is {config}')


    model, global_step = load_model(config, ckpt)

    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")

    logdir = os.path.join(logdir, "samples",f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    run(model,
        imglogdir,
        eta=opt.eta,
        vanilla=opt.vanilla_sample,
        n_samples=opt.n_samples,
        custom_steps=opt.custom_steps,
        batch_size=opt.batch_size,
        nplog=numpylogdir)

    print("done.")
