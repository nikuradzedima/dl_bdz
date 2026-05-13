import os
from datetime import datetime
import comet_ml
import numpy as np
from dotenv import load_dotenv
from src.utils.io_utils import ROOT_PATH


class CometMLWriter:

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        workspace=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        load_dotenv(ROOT_PATH / ".env")
        self.logger = logger
        self.step = 0
        self.mode = "train"
        self.timer = datetime.now()
        self.exp = None
        api_key = os.getenv("COMET_API_KEY")
        if mode == "offline":
            exp_class = comet_ml.OfflineExperiment
            self.exp = exp_class(
                project_name=project_name,
                workspace=workspace,
                experiment_key=run_id,
                log_code=kwargs.get("log_code", False),
            )
        else:
            self.exp = comet_ml.Experiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace or os.getenv("COMET_WORKSPACE") or None,
                experiment_key=run_id,
                log_code=kwargs.get("log_code", False),
                log_graph=kwargs.get("log_graph", False),
                auto_metric_logging=False,
                auto_param_logging=False,
            )
        if run_name:
            self.exp.set_name(run_name)
        self.exp.log_parameters(project_config)

    def set_step(self, step: int, mode: str = "train") -> None:
        self.mode = mode
        previous_step = self.step
        self.step = step
        if previous_step != step:
            duration = datetime.now() - self.timer
            if duration.total_seconds() > 0 and step > previous_step:
                self.add_scalar(
                    "steps_per_sec", (step - previous_step) / duration.total_seconds()
                )
            self.timer = datetime.now()

    def _name(self, name: str) -> str:
        return f"{self.mode}/{name}"

    def add_scalar(self, name: str, value) -> None:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        if self.exp is not None:
            self.exp.log_metric(self._name(name), value, step=self.step)

    def add_scalars(self, scalars: dict) -> None:
        for name, value in scalars.items():
            self.add_scalar(name, value)

    def add_audio(self, name: str, audio, sample_rate: int) -> None:
        if self.exp is None:
            return
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().float().numpy()
        audio = np.asarray(audio)
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        self.exp.log_audio(
            audio_data=audio,
            sample_rate=sample_rate,
            file_name=f"{self._name(name)}.wav",
            step=self.step,
        )

    def add_checkpoint(self, path: str) -> None:
        if self.exp is not None:
            self.exp.log_model("checkpoints", path, overwrite=True)

    def close(self) -> None:
        if self.exp is not None:
            self.exp.end()
