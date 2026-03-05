import os
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（根据实际情况调整）
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到sys.path的最前面
sys.path.insert(0, project_root)


from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import yaml
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from PIL import Image
from utils_yxb import dict2namespace, namespace2dict
import importlib
from torch.utils.data import DataLoader,Dataset




class ImageLogger(pl.Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TensorBoardLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):

        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            # 使用目标处理流程替换原始处理
            grid = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, batch_idx, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")


# 辅助函数保持不变
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if config.__contains__('params'):
        return get_obj_from_str(config["target"])(**vars(config['params']))
    else:
        return get_obj_from_str(config["target"])()


class VQGANdataModule(pl.LightningDataModule):
    def __init__(self, dataconfig):
        super().__init__()
        self.dataconfig = vars(dataconfig)
        if "train" in dataconfig:
            self.tarinconfig = self.dataconfig['train']
        if "val" in dataconfig:
            self.testconfig = self.dataconfig['val']
        if "test" in dataconfig:
            self.valconfig = self.dataconfig['test']

    def prepare_data(self):
        pass

    def setup(self, stage):
        if hasattr(self, "tarinconfig"):
            self.train_dataset = instantiate_from_config(vars(self.tarinconfig))
        if hasattr(self, "valconfig"):
            self.val_dataset = instantiate_from_config(vars(self.valconfig))
        if hasattr(self, "testconfig"):
            self.test_dataset = instantiate_from_config(vars(self.testconfig))

    def train_dataloader(self):
        print(f"加载路径: {self.tarinconfig.params.path}")
        return DataLoader(
            self.train_dataset, batch_size=self.tarinconfig.batch, shuffle=True
        )

    def val_dataloader(self):
        print(f"加载路径: {self.valconfig.params.path}")
        return DataLoader(
            self.val_dataset, batch_size=self.valconfig.batch,shuffle=True
        )




def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('-c', '--config', type=str, default='/mnt/D/chenkang/ldm_pet/configs/autoencoder/vqgan-f4.yaml', help='Path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--gpu_ids', type=str, default='1,', help='gpu ids, 0,1,2,3 cpu=-1')
    parser.add_argument("-r", "--root_dir", type=str, default="/mnt/D/chenkang/ldm_pet",
                        help="root dir")
    parser.add_argument('--max_epoch', type=int, default=None, help='optimizer checkpoint')
    # 添加ImageLogger的参数
    parser.add_argument('--log_img_freq', type=int, default=1000, help='Frequency to log images (in batches)')
    parser.add_argument('--max_log_images', type=int, default=8, help='Maximum number of images to log per batch')
    parser.add_argument("--ckpt", type=str, default=None, help="ckpt to resume or test")
    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    if args.max_epoch is not None:
        namespace_config.training.n_epochs = args.max_epoch

    dict_config = namespace2dict(namespace_config)

    return namespace_config, dict_config


if __name__ == "__main__":
    namespace_config, dict_config = parse_args_and_config()
    model = instantiate_from_config(vars(namespace_config.VQGAN))
    datamodel = instantiate_from_config(vars(namespace_config.data))

    # 保存 结果
    namespace_config.args.log_img_freq = 1
    namespace_config.args.max_log_images = 100000
    namespace_config.args.root_dir = os.path.join(namespace_config.args.root_dir, namespace_config.VQGAN.name)
    model.learning_rate = namespace_config.VQGAN.learning_rate


    image_logger = ImageLogger(
        batch_frequency=namespace_config.args.log_img_freq,
        max_images=namespace_config.args.max_log_images,
        clamp=True,
        increase_log_steps=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/rec_loss",  # 监控的指标（根据实际修改，如val_acc）
        mode="min",  # "min"表示指标越小越好，"max"反之
        save_top_k=1,  # 保存最佳的一个模型
        filename="best-model-{epoch:02d}",  # 文件名格式
        save_last=True,  # 额外保存最后一个epoch的模型（可选）
    )


    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self


    # import torch
    # import torchvision
    # import os
    # from PIL import Image
    #
    # # 1. 将模型移到GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vqgan = VQModel(**vars(namespace_config.VQGAN.params)).eval().to(device)
    # vqgan.train = disabled_train
    # for param in vqgan.parameters():
    #     param.requires_grad = False
    #
    # # 2. 准备数据集和数据加载器
    # datasets = PET2CTdataset_RGB("/home/dpet/data2/yxb/sheng dataset")
    # dataload = DataLoader(dataset=datasets, batch_size=2, shuffle=False)
    #
    # # 3. 创建保存图像的目录
    # os.makedirs("./original_images", exist_ok=True)
    # os.makedirs("./reconstructed_images", exist_ok=True)
    #
    # # 4. 处理每个批次
    # for i, batch in enumerate(dataload):
    #
    #     # 将数据移到GPU
    #     x = batch.to(device)
    #
    #     # 执行重建
    #     with torch.no_grad():
    #         rec = vqgan(x)[0]
    #
    #     # 将数据移回CPU以便保存
    #     x_cpu = x.cpu()
    #     rec_cpu = rec.cpu()
    #
    #     # 保存原始图像
    #     for j in range(x_cpu.size(0)):
    #         # 处理单张图像
    #         img = x_cpu[j]
    #
    #         # 转换为 [0, 255] 范围的 uint8 并调整维度顺序
    #         img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    #         # 创建PIL图像并保存
    #         pil_img = Image.fromarray(img)
    #         filename = f"original_images/batch_{i}_img_{j}.png"
    #         pil_img.save(filename)
    #
    #     # 保存重建图像
    #     for j in range(rec_cpu.size(0)):
    #         # 处理单张图像
    #         img = rec_cpu[j]
    #
    #         # 转换为 [0, 255] 范围的 uint8 并调整维度顺序
    #         img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    #
    #         # 创建PIL图像并保存
    #         pil_img = Image.fromarray(img)
    #         filename = f"reconstructed_images/batch_{i}_img_{j}.png"
    #         pil_img.save(filename)
    #
    #     print(f"Processed batch {i}, saved {x_cpu.size(0)} original and reconstructed images")


    if namespace_config.args.train:
        trainer = pl.Trainer(
            max_epochs=namespace_config.args.max_epoch,
            gpus=namespace_config.args.gpu_ids,
            default_root_dir=namespace_config.args.root_dir,
            check_val_every_n_epoch=1,
            callbacks=[image_logger, checkpoint_callback],  # 添加ImageLogger回调

        )
        if namespace_config.args.ckpt is not None:
            trainer.fit(model, datamodule=datamodel, ckpt_path=namespace_config.args.ckpt)
        else:
            trainer.fit(model, datamodule=datamodel)
    else:
        namespace_config.args.root_dir = os.path.join(namespace_config.args.root_dir, "test")
        trainer = pl.Trainer(
            max_epochs=namespace_config.args.max_epoch,
            gpus=namespace_config.args.gpu_ids,
            default_root_dir=namespace_config.args.root_dir,
            check_val_every_n_epoch=1,
            callbacks=[image_logger, checkpoint_callback],  # 添加ImageLogger回调

        )
        print("------------------------开始测试------------------------")

        trainer.validate(model, datamodule=datamodel, ckpt_path=namespace_config.args.ckpt)
