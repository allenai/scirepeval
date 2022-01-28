from dataclasses import dataclass, field

from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from torch.optim.optimizer import Optimizer

"""Code picked up from fairseq implementation and adapted to suit pytorch lightning"""
@dataclass
class InverseSquareRootScheduleConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=4000,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    warmup_init_lr: float = field(
        default=-1,
        metadata={
            "help": "initial learning rate during warmup phase; default is args.lr"
        },
    )
    lr: float = II("params.optimization.lr")


class InverseSquareRootSchedule:
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, config: InverseSquareRootScheduleConfig, optimizer: Optimizer):
        self.config = config
        self.optimizer = optimizer
        warmup_end_lr = config.lr
        if config.warmup_init_lr < 0:
            config.warmup_init_lr = 0 if config.warmup_updates > 0 else warmup_end_lr

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_lr - config.warmup_init_lr) / config.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * config.warmup_updates ** 0.5

        # initial learning rate
        self.lr = config.warmup_init_lr

    def step(self):
        """Update the learning rate after each update."""
        num_updates = self.optimizer.state[self.optimizer.param_groups[0]["params"][-1]]["step"]
        for g in self.optimizer.param_groups:
            if num_updates < self.config.warmup_updates:
                g['lr'] = self.config.warmup_init_lr + num_updates * self.lr_step
            else:
                g['lr'] = self.decay_factor * num_updates ** -0.5
        self.lr = self.optimizer.param_groups[0]['lr']
        return self.lr
