from dataclasses import dataclass, field

@dataclass
class RecordOursPipelineConfig:
    dataset_path: str
    episode_num: int
    episode_len: int
    recorded_by: str
    task: str
    create_video: bool = True
    use_devices: bool = True
    fps: int = 30
    cam_list: list[str] = field(default_factory=lambda: ['wrist', 'exo', 'table'])