from collections import defaultdict
from pathlib import Path
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from src.datasets.data_utils import inf_loop


class RunningAverage:

    def __init__(self):
        self.values = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, name: str, value, n: int = 1) -> None:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        self.values[name] += float(value) * n
        self.counts[name] += n

    def result(self) -> dict[str, float]:
        return {
            name: self.values[name] / max(self.counts[name], 1) for name in self.values
        }


def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


class SoundStreamTrainer:

    def __init__(
        self,
        model,
        discriminator,
        criterion,
        metrics,
        optimizer_g,
        optimizer_d,
        config,
        device,
        dataloaders,
        logger,
        writer,
        save_dir: Path,
    ):
        self.model = model
        self.discriminator = discriminator
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.config = config
        self.cfg = config.trainer
        self.device = device
        self.train_loader = inf_loop(dataloaders["train"])
        self.eval_loaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.logger = logger
        self.writer = writer
        self.save_dir = save_dir
        self.global_step = 0
        self.best_stoi = float("-inf")
        if self.cfg.get("resume_from") is not None:
            self.load_checkpoint(Path(self.cfg.resume_from))

    def train(self) -> None:
        for epoch in range(1, self.cfg.n_epochs + 1):
            self.train_epoch(epoch)
            if self.global_step % self.cfg.eval_step != 0:
                self.evaluate_all(epoch)
            if self.global_step >= self.cfg.total_steps:
                break
        self.save_checkpoint("model_last.pth")
        self.writer.close()

    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        self.discriminator.train()
        avg = RunningAverage()
        progress = tqdm(
            range(self.cfg.epoch_len),
            desc=f"train epoch {epoch}",
            dynamic_ncols=True,
            mininterval=2.0,
        )
        for _ in progress:
            batch = next(self.train_loader)
            batch = self.move_batch_to_device(batch)
            losses, outputs = self.train_step(batch)
            self.global_step += 1
            for name, value in losses.items():
                avg.update(name, value)
            avg.update("codebook_perplexity", outputs["codebook_perplexity"])
            if self.global_step % self.cfg.log_step == 0:
                scalars = avg.result()
                self.writer.set_step(self.global_step, "train")
                self.writer.add_scalars(scalars)
                progress.set_postfix(
                    generator_loss=f"{scalars.get('generator_loss', 0.0):.4f}",
                    discriminator_loss=f"{scalars.get('discriminator_loss', 0.0):.4f}",
                )
                avg = RunningAverage()
            if self.global_step % self.cfg.audio_log_step == 0:
                self.log_audio(batch, outputs, mode="train")
            if self.global_step % self.cfg.save_step == 0:
                self.save_checkpoint(f"checkpoint-step{self.global_step}.pth")
            if self.global_step % self.cfg.eval_step == 0:
                self.evaluate_all(epoch)
            if self.global_step >= self.cfg.total_steps:
                break

    def train_step(self, batch: dict) -> tuple[dict, dict]:
        audio = batch["audio"]
        outputs = self.model(audio)
        fake_audio = outputs["audio_hat"]
        set_requires_grad(self.discriminator, True)
        self.optimizer_d.zero_grad(set_to_none=True)
        real_d = self.discriminator(audio)
        fake_d = self.discriminator(fake_audio.detach())
        discriminator_loss = self.criterion.discriminator_loss(real_d, fake_d)
        discriminator_loss.backward()
        self.optimizer_d.step()
        set_requires_grad(self.discriminator, False)
        self.optimizer_g.zero_grad(set_to_none=True)
        with torch.no_grad():
            real_g = self.discriminator(audio)
        fake_g = self.discriminator(fake_audio)
        generator_losses = self.criterion.generator_loss(
            real_audio=audio,
            fake_audio=fake_audio,
            real_outputs=real_g,
            fake_outputs=fake_g,
            commitment_loss=outputs["commitment_loss"],
        )
        generator_losses["generator_loss"].backward()
        if self.cfg.get("max_grad_norm") is not None:
            clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer_g.step()
        set_requires_grad(self.discriminator, True)
        losses = {"discriminator_loss": discriminator_loss, **generator_losses}
        return (losses, outputs)

    @torch.no_grad()
    def evaluate_all(self, epoch: int) -> None:
        for part, loader in self.eval_loaders.items():
            logs, last_batch, last_outputs = self.evaluate(part, loader)
            self.writer.set_step(self.global_step, part)
            self.writer.add_scalars(logs)
            self.log_audio(last_batch, last_outputs, mode=part)
            log_line = ", ".join((f"{k}: {v:.4f}" for k, v in logs.items()))
            self.logger.info(
                f"epoch {epoch} step {self.global_step} {part}: {log_line}"
            )
            stoi = logs.get("STOI")
            if part == "test" and stoi is not None and (stoi > self.best_stoi):
                self.best_stoi = stoi
                self.save_checkpoint("model_best.pth")

    @torch.no_grad()
    def evaluate(self, part: str, loader) -> tuple[dict[str, float], dict, dict]:
        self.model.eval()
        self.discriminator.eval()
        avg = RunningAverage()
        last_batch = None
        last_outputs = None
        max_batches = self.cfg.get("max_eval_batches")
        total_batches = len(loader)
        if max_batches is not None:
            total_batches = min(total_batches, int(max_batches))
        progress_interval = max(1, total_batches // 10)
        self.logger.info(f"eval {part}: считаем {total_batches} batches")
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = self.move_batch_to_device(batch)
            outputs = self.model(batch["audio"])
            real = self.discriminator(batch["audio"])
            fake = self.discriminator(outputs["audio_hat"])
            d_loss = self.criterion.discriminator_loss(real, fake)
            g_losses = self.criterion.generator_loss(
                real_audio=batch["audio"],
                fake_audio=outputs["audio_hat"],
                real_outputs=real,
                fake_outputs=fake,
                commitment_loss=outputs["commitment_loss"],
            )
            avg.update("discriminator_loss", d_loss)
            for name, value in g_losses.items():
                avg.update(name, value)
            avg.update("codebook_perplexity", outputs["codebook_perplexity"])
            for metric in self.metrics.get("inference", []):
                avg.update(metric.name, metric(**batch, **outputs))
            last_batch = batch
            last_outputs = outputs
            done_batches = batch_idx + 1
            if done_batches == total_batches or done_batches % progress_interval == 0:
                self.logger.info(f"eval {part}: {done_batches}/{total_batches} batches")
        self.model.train()
        self.discriminator.train()
        return (avg.result(), last_batch, last_outputs)

    def move_batch_to_device(self, batch: dict) -> dict:
        for key in self.cfg.device_tensors:
            if key in batch:
                batch[key] = batch[key].to(self.device)
        return batch

    def log_audio(self, batch: dict | None, outputs: dict | None, mode: str) -> None:
        if batch is None or outputs is None:
            return
        self.writer.set_step(self.global_step, mode)
        self.writer.add_audio("original", batch["audio"][0], self.cfg.sample_rate)
        self.writer.add_audio(
            "reconstructed", outputs["audio_hat"][0], self.cfg.sample_rate
        )

    def checkpoint_state(self) -> dict:
        return {
            "step": self.global_step,
            "model": self.model.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "config": self.config,
            "best_stoi": self.best_stoi,
        }

    def save_checkpoint(self, name: str) -> None:
        path = self.save_dir / name
        torch.save(self.checkpoint_state(), path)
        self.logger.info(f"checkpoint сохранен: {path}")
        if self.config.writer.get("log_checkpoints", False):
            self.writer.add_checkpoint(str(path))

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        self.global_step = checkpoint.get("step", 0)
        self.best_stoi = checkpoint.get("best_stoi", float("-inf"))
        self.logger.info(f"checkpoint загружен из {path}, step {self.global_step}")
