"""RenderConfig — every knob in one dataclass.

JSON / YAML round-trip is supported via :meth:`from_file` / :meth:`to_file`.
Defaults are tuned for "looks reasonable on a 1080p frame with the CUDA
backend" — you tune from there.

The dataclass is intentionally flat (no nested config records) to keep
serialization trivial. Sub-configs live as small dataclasses next to it
when they're large enough to warrant grouping (gsplat, lod, completion).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

AAMode = Literal["off", "fxaa", "taa", "smaa"]
ShadowMode = Literal["off", "csm", "vsm"]
NeuralUpscale = Literal["none", "fsr", "dlss"]
NeuralRelight = Literal["none", "ssgi", "neural_ibl"]
ColorSpace = Literal["linear", "sRGB"]
OutputDtype = Literal["uint8", "float16", "float32"]
DeviceHint = Literal["auto", "cuda", "wgpu", "cpu", "mps"]


@dataclass(slots=True)
class GsplatConfig:
    enabled: bool = True
    sigma_scale: float = 1.0
    densify: bool = True
    densify_grad_threshold: float = 0.0002
    sh_degree: int = 3            # spherical-harmonic order


@dataclass(slots=True)
class LodConfig:
    enabled: bool = True
    screen_space_error_px: float = 1.5
    max_chunks_in_vram: int = 256


@dataclass(slots=True)
class CompletionConfig:
    enabled: bool = False
    mlp_width: int = 64
    mlp_depth: int = 3
    hash_levels: int = 16


@dataclass(slots=True)
class SurfelConfig:
    enabled: bool = True
    radius_factor: float = 1.5    # factor of k-NN spacing


@dataclass(slots=True)
class FogConfig:
    enabled: bool = False
    density: float = 0.02
    color: tuple[float, float, float] = (0.7, 0.78, 0.86)
    height_falloff: float = 0.1


@dataclass(slots=True)
class RenderConfig:
    # ---- output --------------------------------------------------------
    width: int = 1280
    height: int = 720
    samples: int = 1
    aa: AAMode = "fxaa"
    output_dtype: OutputDtype = "float32"
    output_color_space: ColorSpace = "linear"
    sensor_outputs: tuple[str, ...] = ("rgb",)
    # ---- device --------------------------------------------------------
    device: DeviceHint = "auto"
    vram_budget_mb: float = 4096.0
    # ---- determinism ---------------------------------------------------
    seed: int = 0
    # ---- rendering toggles --------------------------------------------
    shadows: ShadowMode = "csm"
    shadow_map_resolution: int = 512        # per-cascade shadow map size (px)
    shadow_bias_constant: float = 0.2       # depth bias, constant term (texels)
    shadow_bias_slope: float = 1.0          # depth bias, slope-term multiplier (x clamped ground slope)
    shadow_bias_override: float | None = None   # explicit world-space bias; wins over constant+slope
    # ---- anti-aliasing -------------------------------------------------
    # ssaa = full-scene supersampling: geometry renders at ssaa× the output
    # resolution and is area-averaged down before post passes. 1 = off.
    ssaa: int = 1
    # TAA (aa="taa"): Halton-2,3 sub-pixel jitter + exponential history
    # blend with 3x3 neighbourhood clamping. History lives on the TaaPass
    # instance and resets when camera / resolution / jitter phase changes.
    taa_alpha: float = 0.1            # weight of the CURRENT frame (0, 1]
    taa_jitter_frames: int = 8        # jitter sequence period (frames)
    # ---- bloom / glow --------------------------------------------------
    bloom: bool = True
    bloom_threshold: float = 1.0            # HDR knee; only brighter pixels glow
    bloom_intensity: float = 0.6            # additive strength of the halo
    bloom_radius: int = 1                   # gaussian pyramid levels (1 = legacy 5-tap)
    # ---- god rays (crepuscular rays from the sun) ----------------------
    god_rays: bool = False
    god_ray_intensity: float = 0.6
    god_ray_samples: int = 24               # radial-blur taps per pixel
    god_ray_decay: float = 0.95             # per-tap falloff away from the sun
    # ---- transparency ---------------------------------------------------
    # Alpha-blended rendering for materials with alpha < 1 (glass). Two-pass
    # draw: opaque first (depth write), then transparent sorted back-to-front
    # (depth test, no depth write). Off = legacy fully-opaque behavior.
    transparency: bool = False
    exposure: float = 1.0
    # ---- point clouds --------------------------------------------------
    gsplat: GsplatConfig = field(default_factory=GsplatConfig)
    surfels: SurfelConfig = field(default_factory=SurfelConfig)
    lod: LodConfig = field(default_factory=LodConfig)
    completion: CompletionConfig = field(default_factory=CompletionConfig)
    # ---- volumes -------------------------------------------------------
    fog: FogConfig = field(default_factory=FogConfig)
    # ---- simulation (water animation, particles) -----------------------
    simulation_dt: float = 1.0 / 60.0   # per-render frame step, seconds
    # ---- neural FX -----------------------------------------------------
    neural_denoise: bool = False
    # Spatial upscaler: geometry renders at 1/upscale_factor of the output
    # resolution and the upscale pass resolves to full size before tonemap.
    #   "fsr"  — AMD FSR 1.0 (EASU + RCAS), pure torch, runs everywhere
    #   "dlss" — NVIDIA DLSS via an NGX bridge DLL (BONAFIDE_DLSS_DLL);
    #            falls back to FSR with a skip note when unavailable
    neural_upscale: NeuralUpscale = "none"
    upscale_factor: float = 2.0       # output px per internal px, [1, 4]
    upscale_sharpness: float = 0.2    # RCAS sharpening strength, [0, 1]
    neural_relight: NeuralRelight = "none"
    ssgi_intensity: float = 0.5       # indirect bounce gain for "ssgi"
    # ---- differentiable ------------------------------------------------
    differentiable: bool = False
    # ---- profiling -----------------------------------------------------
    profile: bool = False

    # ------------------------------------------------------------ IO
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RenderConfig:
        # Materialize sub-configs even when they came in as plain dicts.
        sub_specs = {
            "gsplat":     GsplatConfig,
            "surfels":    SurfelConfig,
            "lod":        LodConfig,
            "completion": CompletionConfig,
            "fog":        FogConfig,
        }
        kwargs = dict(data)
        for k, sub_cls in sub_specs.items():
            if k in kwargs and isinstance(kwargs[k], dict):
                kwargs[k] = sub_cls(**kwargs[k])
        if "sensor_outputs" in kwargs and isinstance(kwargs["sensor_outputs"], list):
            kwargs["sensor_outputs"] = tuple(kwargs["sensor_outputs"])
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Raise ConfigurationError if any field combination is illegal."""
        from ironengine_bonafide.errors import ConfigurationError
        if self.width <= 0 or self.height <= 0:
            raise ConfigurationError(
                f"width/height must be positive (got {self.width}x{self.height})"
            )
        if self.samples not in (1, 2, 4, 8):
            raise ConfigurationError(f"samples must be 1/2/4/8 (got {self.samples})")
        if self.exposure <= 0.0:
            raise ConfigurationError(f"exposure must be > 0 (got {self.exposure})")
        if self.vram_budget_mb <= 0.0:
            raise ConfigurationError(
                f"vram_budget_mb must be > 0 (got {self.vram_budget_mb})"
            )
        valid_outputs = {"rgb", "depth", "normals", "ids", "albedo"}
        bad = set(self.sensor_outputs) - valid_outputs
        if bad:
            raise ConfigurationError(
                f"sensor_outputs contains invalid keys {sorted(bad)} "
                f"(allowed: {sorted(valid_outputs)})"
            )
        if self.aa not in ("off", "fxaa", "taa", "smaa"):
            raise ConfigurationError(f"aa='{self.aa}' must be off|fxaa|taa|smaa")
        if self.ssaa not in (1, 2, 4):
            raise ConfigurationError(f"ssaa must be 1/2/4 (got {self.ssaa})")
        if self.bloom_threshold < 0.0:
            raise ConfigurationError(f"bloom_threshold must be >= 0 (got {self.bloom_threshold})")
        if self.bloom_intensity < 0.0:
            raise ConfigurationError(f"bloom_intensity must be >= 0 (got {self.bloom_intensity})")
        if self.bloom_radius not in (1, 2, 3, 4):
            raise ConfigurationError(f"bloom_radius must be 1-4 (got {self.bloom_radius})")
        if self.god_ray_intensity < 0.0:
            raise ConfigurationError(f"god_ray_intensity must be >= 0 (got {self.god_ray_intensity})")
        if not (1 <= self.god_ray_samples <= 128):
            raise ConfigurationError(f"god_ray_samples must be 1-128 (got {self.god_ray_samples})")
        if not (0.0 < self.god_ray_decay <= 1.0):
            raise ConfigurationError(f"god_ray_decay must be in (0, 1] (got {self.god_ray_decay})")
        if self.shadows not in ("off", "csm", "vsm"):
            raise ConfigurationError(f"shadows='{self.shadows}' must be off|csm|vsm")
        if self.shadow_map_resolution <= 0:
            raise ConfigurationError(
                f"shadow_map_resolution must be > 0 (got {self.shadow_map_resolution})"
            )
        if self.shadow_bias_constant < 0.0 or self.shadow_bias_slope < 0.0:
            raise ConfigurationError("shadow_bias_constant/slope must be >= 0")
        if self.shadow_bias_override is not None and self.shadow_bias_override < 0.0:
            raise ConfigurationError("shadow_bias_override must be >= 0")
        if self.output_color_space not in ("linear", "sRGB"):
            raise ConfigurationError(
                f"output_color_space='{self.output_color_space}' must be linear|sRGB"
            )
        if self.neural_upscale not in ("none", "fsr", "dlss"):
            raise ConfigurationError(f"neural_upscale='{self.neural_upscale}' invalid")
        if not (1.0 <= self.upscale_factor <= 4.0):
            raise ConfigurationError(
                f"upscale_factor must be in [1, 4] (got {self.upscale_factor})"
            )
        if not (0.0 <= self.upscale_sharpness <= 1.0):
            raise ConfigurationError(
                f"upscale_sharpness must be in [0, 1] (got {self.upscale_sharpness})"
            )
        if self.neural_upscale != "none" and self.upscale_factor > 1.0 and self.ssaa > 1:
            raise ConfigurationError(
                "ssaa > 1 and neural_upscale are mutually exclusive "
                "(both change the internal render resolution)"
            )
        if not (0.0 < self.taa_alpha <= 1.0):
            raise ConfigurationError(f"taa_alpha must be in (0, 1] (got {self.taa_alpha})")
        if self.taa_jitter_frames < 2:
            raise ConfigurationError(
                f"taa_jitter_frames must be >= 2 (got {self.taa_jitter_frames})"
            )
        if self.ssgi_intensity < 0.0:
            raise ConfigurationError(
                f"ssgi_intensity must be >= 0 (got {self.ssgi_intensity})"
            )
        if self.simulation_dt <= 0.0:
            raise ConfigurationError(
                f"simulation_dt must be > 0 (got {self.simulation_dt})"
            )
        if self.neural_relight not in ("none", "ssgi", "neural_ibl"):
            raise ConfigurationError(f"neural_relight='{self.neural_relight}' invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_file(cls, path: str | Path) -> RenderConfig:
        from ironengine_bonafide.errors import ConfigurationError
        text = Path(path).read_text(encoding="utf-8")
        if str(path).endswith((".yaml", ".yml")):
            try:
                import yaml
                data = yaml.safe_load(text)
            except ImportError as exc:
                raise ConfigurationError(
                    "PyYAML required to load YAML configs"
                ) from exc
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def to_file(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
