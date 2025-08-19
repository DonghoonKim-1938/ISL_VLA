import torch
from torch.utils.data.distributed import DistributedSampler
from common.datasets.sampler import EpisodeAwareSampler


def make_dataloader(cfg, dataset, device):
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
        dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    if getattr(cfg, "dist_mode", "ddp") in ("ddp", "fsdp"):
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=cfg.seed, drop_last=False)
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    return dataloader