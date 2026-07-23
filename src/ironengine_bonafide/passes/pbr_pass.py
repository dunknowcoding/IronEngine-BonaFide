"""Mesh PBR pass with CSM shadowing, texture maps and image-based lighting.

Pipeline per mesh:
  1. Backend rasterizes a GBuffer (albedo, world-position, normal, uv,
     tangent, depth, mask).
     - CUDA path: ``nvdiffrast`` deferred shading (no uv/tangent channel
       yet — texture maps are sampled on the CPU reference path only).
     - CPU path:  ``CpuBackend.raster_mesh_gbuffer`` (vectorized,
       perspective-correct).
  2. Material texture maps (``albedo_map`` / ``normal_map`` /
     ``metallic_roughness_map`` / ``ao_map``) are loaded, sRGB-decoded
     where appropriate, and bilinear-sampled at the interpolated UVs.
     Normal maps are applied in tangent space (per-vertex tangents solved
     from the UV Jacobian).
  3. ``_shade_gbuffer`` evaluates each light with a Cook-Torrance GGX BRDF
     (GGX/Trowbridge-Reitz D, Smith Schlick-GGX G, Schlick Fresnel) driven
     by per-pixel ``roughness`` / ``metallic`` and ``emissive``:
     - Directional lights with shadow maps PCF-sample the matching cascade.
     - Point/spot/area lights: simple inverse-square / cone falloff.
     - IBL: diffuse irradiance from the coarsest equirect mip sampled by
       the normal, specular from a roughness-blended mip sample along the
       reflection direction, scaled by the Fresnel-weighted F0. Without an
       IBL a hemisphere ambient is used instead.
  4. ``_composite`` writes the result into ``targets`` with depth testing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ironengine_bonafide.backends.cpu.backend import CpuBackend
from ironengine_bonafide.backends.torch_raster import (
    vertex_normals,
    vertex_tangents,
)
from ironengine_bonafide.core.color import srgb_to_linear
from ironengine_bonafide.core.envmap import (
    build_mip_chain,
    equirect_sample,
    sample_mip_blended,
)
from ironengine_bonafide.core.light import (
    IBL,
    AreaLight,
    DirectionalLight,
    PointLight,
    SpotLight,
)
from ironengine_bonafide.core.shadow import ShadowMap, pcf_sample
from ironengine_bonafide.logging import logger
from ironengine_bonafide.passes.base import PassContext, RenderPass


class PbrPass(RenderPass):
    name = "pbr"

    def required_capabilities(self) -> tuple[str, ...]:
        return ("raster",)

    def is_active(self, ctx: PassContext) -> bool:
        return bool(ctx.scene.meshes)

    def run(self, ctx: PassContext) -> None:
        for mesh_id, mesh in enumerate(ctx.scene.meshes, start=1):
            self._render_one(ctx, mesh, mesh_id)

    # --------------------------------------------------------- per-mesh
    def _render_one(self, ctx: PassContext, mesh, instance_id: int) -> None:    # type: ignore[no-untyped-def]
        h, w, _ = ctx.targets.rgb.shape
        device = ctx.backend.device
        view_proj = ctx.camera.view_proj_torch(ctx.aspect, device=device)
        positions = mesh.positions.to(device)
        indices = mesh.indices.to(device)
        normals = mesh.normals.to(device) if mesh.normals is not None else None
        uvs = mesh.uvs.to(device) if mesh.uvs is not None else None
        if mesh.colors is not None:
            colors = mesh.colors.to(device)
        else:
            albedo = torch.tensor(mesh.material.albedo, device=device, dtype=torch.float32)
            colors = albedo.unsqueeze(0).expand(positions.shape[0], 3).contiguous()

        maps = _MaterialMaps.load(mesh.material)

        # ---- GBuffer raster ----------------------------------------
        if ctx.backend.supports("nvdiffrast"):
            from ironengine_bonafide.backends.cuda.raster import render_mesh_gbuffer
            if normals is None:
                normals = vertex_normals(positions, indices)
            albedo, nrm, depth, mask = render_mesh_gbuffer(
                positions, indices, normals, colors,
                view_proj, w, h,
                background=(0.0, 0.0, 0.0),
            )
            world_pos = _reconstruct_world_pos(depth, view_proj, w, h)
            if maps.any:
                ctx.skipped.append("pbr:texture_maps_cpu_path_only")
            maps = _MaterialMaps()                              # unsupported here
            uv_buf = tan_buf = None
        else:
            cpu = ctx.backend if isinstance(ctx.backend, CpuBackend) else CpuBackend()
            tangents = None
            if maps.normal is not None and uvs is not None:
                if normals is None:
                    normals = vertex_normals(positions, indices)
                tangents = vertex_tangents(positions, indices, uvs, normals)
            gb = cpu.raster_mesh_gbuffer(
                positions=positions.cpu(), indices=indices.cpu(),
                colors=colors.cpu(),
                normals=normals.cpu() if normals is not None else None,
                view_proj=view_proj.cpu(),
                width=w, height=h,
                uvs=uvs.cpu() if (uvs is not None and maps.any) else None,
                tangents=tangents.cpu() if tangents is not None else None,
            )
            out_device = ctx.targets.rgb.device
            albedo = gb.albedo.to(out_device)
            world_pos = gb.world_pos.to(out_device)
            nrm = gb.normal.to(out_device)
            depth = gb.depth.to(out_device)
            mask = gb.mask.to(out_device)
            uv_buf = gb.uv.to(out_device) if gb.uv is not None else None
            tan_buf = gb.tangent.to(out_device) if gb.tangent is not None else None

        # ---- Texture maps (need interpolated UVs) -------------------
        rough_t = metal_t = ao_t = None
        if maps.any:
            if uv_buf is None:
                ctx.skipped.append(f"pbr:{getattr(mesh, 'name', 'mesh')}_maps_no_uvs")
            else:
                if maps.albedo is not None:
                    albedo = albedo * _sample_uv(maps.albedo.to(albedo.device), uv_buf)
                if maps.normal is not None and tan_buf is not None:
                    nrm = _apply_normal_map(nrm, tan_buf,
                                            maps.normal.to(nrm.device), uv_buf)
                if maps.metallic_roughness is not None:
                    mr = _sample_uv(maps.metallic_roughness.to(albedo.device), uv_buf)
                    rough_t = (mr[..., 1] * float(mesh.material.roughness)).clamp(0.045, 1.0)
                    metal_t = (mr[..., 2] * float(mesh.material.metallic)).clamp(0.0, 1.0)
                if maps.ao is not None:
                    ao_t = _sample_uv(maps.ao.to(albedo.device), uv_buf)[..., 0].clamp(0.0, 1.0)

        # ---- Shade -------------------------------------------------
        shadow_maps: list[ShadowMap] = getattr(ctx.targets, "shadow_maps", []) or []
        view_depth = None
        if shadow_maps:
            # Positive camera-space distance per fragment for CSM cascade
            # selection (view space looks down -Z).
            vm = ctx.camera.view_matrix()               # type: ignore[attr-defined]
            fwd = torch.tensor(vm[2, :3], device=world_pos.device,
                               dtype=world_pos.dtype)
            view_depth = -(world_pos @ fwd + float(vm[2, 3]))
        shaded = _shade_gbuffer(
            albedo=albedo, world_pos=world_pos, normals=nrm,
            lights=ctx.scene.lights, ibl=ctx.scene.ibl, mask=mask,
            shadow_maps=shadow_maps,
            material=mesh.material,
            cam_pos=_camera_position(ctx.camera, albedo.device, albedo.dtype),
            rough=rough_t, metal=metal_t, ao=ao_t,
            view_depth=view_depth,
        )
        self._composite(ctx, shaded, depth, nrm, instance_id, mask, colors_albedo=albedo)

    # --------------------------------------------------------- composite
    def _composite(self, ctx, shaded, depth, nrm, instance_id, mask, *, colors_albedo):  # type: ignore[no-untyped-def]
        better = (depth < ctx.targets.depth) & (mask > 0.5)
        if not torch.any(better):
            return
        ctx.targets.rgb[better] = shaded[better]
        ctx.targets.depth[better] = depth[better]
        ctx.targets.normals[better] = nrm[better]
        ctx.targets.albedo[better] = colors_albedo[better]
        ctx.targets.ids[better] = int(instance_id)


# --------------------------------------------------------------- shading
# Hemisphere ambient: sky above, bounce tint below (fallback when no IBL).
_SKY_COLOR = (1.0, 1.0, 1.0)
_GROUND_COLOR = (0.35, 0.33, 0.30)
_AMBIENT_INTENSITY = 0.25
# Dielectric Fresnel baseline (ior ≈ 1.45 → ((ior-1)/(ior+1))² ≈ 0.04).
_F0_DIELECTRIC = 0.04


def _camera_position(camera, device, dtype) -> torch.Tensor:              # type: ignore[no-untyped-def]
    """World-space eye position for Perspective/Orthographic/Sensor cameras."""
    if hasattr(camera, "position"):
        return torch.tensor(camera.position, device=device, dtype=dtype)
    pose = getattr(camera, "pose", None)
    if pose is not None:
        return torch.as_tensor(pose[:3, 3]).to(device=device, dtype=dtype)
    return torch.zeros(3, device=device, dtype=dtype)


def _ggx_brdf(albedo, nrm, view, ldir, radiance, f0, metallic, alpha):    # type: ignore[no-untyped-def]
    """Cook-Torrance GGX microfacet BRDF for one light.

    ``ldir`` is (3,) or HxWx3, ``radiance`` is HxWx3 (color·intensity·
    attenuation·visibility). ``metallic`` / ``alpha`` may be per-pixel
    (H, W) maps. Energy-conserving: kd = (1-F)(1-metallic).
    """
    ndotl = (nrm * ldir).sum(dim=-1).clamp(min=0.0).unsqueeze(-1)
    ndotv = (nrm * view).sum(dim=-1).clamp(min=0.0).unsqueeze(-1)
    half = ldir + view
    half = half / torch.linalg.norm(half, dim=-1, keepdim=True).clamp(min=1e-9)
    ndoth = (nrm * half).sum(dim=-1).clamp(min=0.0).unsqueeze(-1)
    vdoth = (view * half).sum(dim=-1).clamp(0.0, 1.0).unsqueeze(-1)

    # D — GGX / Trowbridge-Reitz
    a2 = alpha * alpha
    d_denom = ndoth * ndoth * (a2 - 1.0) + 1.0
    distr = a2 / (math.pi * d_denom * d_denom + 1e-9)

    # G — Smith Schlick-GGX (direct-lighting remapping)
    k = (alpha + 1.0) ** 2 / 8.0
    geom = (_schlick_g(ndotl, k) * _schlick_g(ndotv, k))

    # F — Schlick
    fres = f0 + (1.0 - f0) * (1.0 - vdoth).pow(5)

    spec = (distr * geom * fres) / (4.0 * ndotl * ndotv + 1e-4)
    kd = (1.0 - fres) * (1.0 - metallic)
    return (kd * albedo + spec) * radiance * ndotl


def _schlick_g(cos_theta: torch.Tensor, k: torch.Tensor | float) -> torch.Tensor:
    return cos_theta / (cos_theta * (1.0 - k) + k + 1e-9)


# ------------------------------------------------------------- IBL caching
# Mip chains are built once per (IBL object, device) and reused — the
# envmap is static for a scene, so this keeps per-frame cost to two
# equirect lookups.
_IBL_MIP_CACHE: dict[tuple[int, str], list[torch.Tensor]] = {}


def _ibl_mips(ibl: IBL, device: torch.device | str, dtype: torch.dtype) -> list[torch.Tensor] | None:
    key = (id(ibl), str(device))
    if key in _IBL_MIP_CACHE:
        return _IBL_MIP_CACHE[key]
    try:
        pixels = torch.as_tensor(
            np.ascontiguousarray(ibl.load()[..., :3]), dtype=torch.float32,
        ).to(device=device, dtype=dtype)
    except Exception as exc:                                    # noqa: BLE001
        logger.warning(f"IBL load failed ({exc}); using hemisphere ambient")
        return None
    if pixels.ndim != 3 or pixels.shape[0] < 2 or pixels.shape[1] < 2:
        return None
    mips = build_mip_chain(pixels)
    _IBL_MIP_CACHE[key] = mips
    return mips


def _shade_gbuffer(
    *, albedo, world_pos, normals, lights, ibl, mask, shadow_maps,
    material=None, cam_pos=None, rough=None, metal=None, ao=None,
    view_depth=None,
):                                                                       # type: ignore[no-untyped-def]
    """Apply Cook-Torrance GGX lights + ambient/IBL + CSM shadows to a
    GBuffer (HxWx3 each). ``material`` supplies scalar roughness /
    metallic / emissive; ``rough`` / ``metal`` / ``ao`` are optional
    per-pixel (H, W) overrides from texture maps. ``cam_pos`` is the
    world-space eye position. ``view_depth`` is the optional per-pixel (H, W)
    positive camera-space distance used for CSM cascade selection."""
    device, dtype = albedo.device, albedo.dtype
    h, w, _ = albedo.shape
    if rough is None:
        rough = torch.full((h, w), min(1.0, max(0.045, float(getattr(material, "roughness", 0.7)))),
                           device=device, dtype=dtype)
    if metal is None:
        metal = torch.full((h, w), min(1.0, max(0.0, float(getattr(material, "metallic", 0.0)))),
                           device=device, dtype=dtype)
    alpha = (rough * rough).unsqueeze(-1)             # (H, W, 1)
    metal3 = metal.unsqueeze(-1)
    f0 = albedo * metal3 + (1.0 - metal3) * _F0_DIELECTRIC
    if cam_pos is None:
        cam_pos = torch.zeros(3, device=device, dtype=dtype)
    view = cam_pos - world_pos
    view = view / torch.linalg.norm(view, dim=-1, keepdim=True).clamp(min=1e-9)

    # ---- Ambient: real IBL sampling, or hemisphere fallback ---------
    mips = _ibl_mips(ibl, device, dtype) if ibl is not None else None
    if mips is not None and ibl is not None:
        intensity = float(getattr(ibl, "intensity", 1.0))
        # Diffuse irradiance ≈ coarsest mip sampled along the normal.
        irradiance = equirect_sample(mips[-1], normals)
        # Specular: reflection direction, roughness-blended mip level.
        ndotv = (normals * view).sum(dim=-1, keepdim=True).clamp(min=0.0)
        refl = 2.0 * ndotv * normals - view
        lvl = rough * (len(mips) - 1)
        spec_env = sample_mip_blended(mips, refl, lvl)
        fres_env = f0 + (1.0 - f0) * (1.0 - ndotv).pow(5)
        kd_env = (1.0 - metal3)
        out = (kd_env * albedo * irradiance + fres_env * spec_env) * intensity
    else:
        # Hemisphere ambient — sky for up-facing normals, ground tint below.
        sky = torch.tensor(_SKY_COLOR, device=device, dtype=dtype)
        ground = torch.tensor(_GROUND_COLOR, device=device, dtype=dtype)
        hemi = (0.5 + 0.5 * normals[..., 1]).clamp(0.0, 1.0).unsqueeze(-1)
        out = albedo * (ground + (sky - ground) * hemi) * _AMBIENT_INTENSITY
    if ao is not None:
        out = out * ao.unsqueeze(-1)

    for lt in lights:
        if isinstance(lt, DirectionalLight):
            ldir = -torch.tensor(lt.direction, device=device, dtype=dtype)
            ldir = ldir / (torch.linalg.norm(ldir) + 1e-9)
            col = torch.tensor(lt.color, device=device, dtype=dtype)
            radiance = (col * lt.intensity).expand(albedo.shape)

            if lt.cast_shadow and shadow_maps:
                visibility = _shadow_factor_csm(world_pos, shadow_maps,
                                                view_depth=view_depth,
                                                normals=normals,
                                                light_dir=ldir)
                radiance = radiance * visibility.unsqueeze(-1)
            out = out + _ggx_brdf(albedo, normals, view, ldir, radiance, f0, metal3, alpha)

        elif isinstance(lt, PointLight):
            light_pos = torch.tensor(lt.position, device=device, dtype=dtype)
            to_light = light_pos - world_pos
            dist = torch.linalg.norm(to_light, dim=-1, keepdim=True).clamp(min=1e-3)
            ldir = to_light / dist
            atten = (1.0 - (dist / lt.range).clamp(0.0, 1.0)).pow(2)
            col = torch.tensor(lt.color, device=device, dtype=dtype)
            radiance = col * (lt.intensity * atten)
            out = out + _ggx_brdf(albedo, normals, view, ldir, radiance, f0, metal3, alpha)

        elif isinstance(lt, SpotLight):
            light_pos = torch.tensor(lt.position, device=device, dtype=dtype)
            spot_dir = torch.tensor(lt.direction, device=device, dtype=dtype)
            spot_dir = spot_dir / (torch.linalg.norm(spot_dir) + 1e-9)
            to_light = light_pos - world_pos
            dist = torch.linalg.norm(to_light, dim=-1, keepdim=True).clamp(min=1e-3)
            ldir = to_light / dist
            cos_inner = float(torch.cos(torch.tensor(lt.inner_deg).deg2rad()))
            cos_outer = float(torch.cos(torch.tensor(lt.outer_deg).deg2rad()))
            cos_l = (-ldir * spot_dir).sum(dim=-1).clamp(min=0.0).unsqueeze(-1)
            spot_factor = ((cos_l - cos_outer) / max(1e-3, cos_inner - cos_outer)).clamp(0.0, 1.0)
            atten = (1.0 - (dist / lt.range).clamp(0.0, 1.0)).pow(2)
            col = torch.tensor(lt.color, device=device, dtype=dtype)
            radiance = col * (lt.intensity * spot_factor * atten)
            out = out + _ggx_brdf(albedo, normals, view, ldir, radiance, f0, metal3, alpha)

        elif isinstance(lt, AreaLight):
            light_pos = torch.tensor(lt.position, device=device, dtype=dtype)
            to_light = light_pos - world_pos
            dist = torch.linalg.norm(to_light, dim=-1, keepdim=True).clamp(min=1e-3)
            ldir = to_light / dist
            col = torch.tensor(lt.color, device=device, dtype=dtype)
            radiance = col * lt.intensity
            out = out + _ggx_brdf(albedo, normals, view, ldir, radiance, f0, metal3, alpha)

        elif isinstance(lt, IBL):
            # IBL handled above; skip here to avoid double-counting.
            continue

    emissive = getattr(material, "emissive", None)
    if emissive is not None and any(float(c) != 0.0 for c in emissive):
        out = out + torch.tensor(emissive, device=device, dtype=dtype)

    return out * mask.unsqueeze(-1)


def _shadow_factor_csm(world_pos: torch.Tensor, shadow_maps: list[ShadowMap],
                       view_depth: torch.Tensor | None = None,
                       normals: torch.Tensor | None = None,
                       light_dir: torch.Tensor | None = None) -> torch.Tensor:
    """Returns visibility ∈ [0, 1]; 1 = fully lit.

    The receiver-side depth bias travels with each map
    (``ShadowMap.receiver_bias_ndc``, config-driven, in light NDC) — there
    is no hidden hard-coded bias here. When ``normals`` + ``light_dir`` are
    supplied (the production GBuffer path), a per-fragment slope-scaled
    receiver term tops the bias up for receivers steeper than the
    horizontal-ground slope already baked into the depth map
    (``ShadowMap.slope_scale`` / ``ShadowMap.slope_tan_ref``) — grazing-lit
    walls need several texels more bias than floors and otherwise acne.
    When ``view_depth`` (positive
    camera-space distance per fragment, HxW) is supplied, each fragment is
    evaluated only against the cascade whose ``z_split_near``/``z_split_far``
    range contains it; otherwise the first cascade whose light-space AABB
    contains the fragment wins. Fragments outside any cascade stay lit."""
    h, w, _ = world_pos.shape
    visibility = torch.ones((h, w), device=world_pos.device, dtype=world_pos.dtype)
    flat = world_pos.reshape(-1, 3)
    ones = torch.ones((flat.shape[0], 1), device=flat.device, dtype=flat.dtype)
    homog = torch.cat([flat, ones], dim=1)
    vd_flat = view_depth.reshape(-1) if view_depth is not None else None
    nrm_flat = None
    if normals is not None and light_dir is not None:
        nrm_flat = normals.reshape(-1, 3)
        nrm_flat = nrm_flat / (torch.linalg.norm(nrm_flat, dim=-1, keepdim=True) + 1e-9)

    for sm in shadow_maps:
        clip = homog @ sm.light_view_proj.T
        ndc = clip[:, :3] / clip[:, 3:4].clamp(min=1e-6)
        in_cascade = ((ndc[:, 0].abs() <= 1.0)
                      & (ndc[:, 1].abs() <= 1.0)
                      & (ndc[:, 2].abs() <= 1.0))
        if vd_flat is not None:
            eps = max(0.5 * float(getattr(sm, "texel_size_world", 0.0)), 1e-6)
            in_cascade = in_cascade & (vd_flat >= float(sm.z_split_near) - eps) \
                                    & (vd_flat < float(sm.z_split_far) + eps)
        if not torch.any(in_cascade):
            continue
        uv = (ndc[:, :2] * 0.5 + 0.5)
        cur_z = ndc[:, 2]
        bias = float(getattr(sm, "receiver_bias_ndc", 0.0) or 0.0)
        pcf_kw: dict = {}
        if nrm_flat is not None and getattr(sm, "slope_scale", 0.0) > 0.0:
            pcf_kw = dict(
                normals=nrm_flat, light_dir=light_dir,
                texel_size_world=float(sm.texel_size_world),
                ndc_per_world=float(sm.ndc_per_world),
                slope_scale=float(sm.slope_scale),
                # Match the ground-slope clamp (core.shadow tan_max=8) so
                # near-parallel receivers get the full correction they need.
                slope_tan_max=8.0,
                slope_tan_ref=float(getattr(sm, "slope_tan_ref", 0.0)),
            )
        lit = pcf_sample(sm.depth, uv, cur_z, bias=bias, radius=1, **pcf_kw)
        # First cascade wins per-pixel
        vis_flat = visibility.flatten()
        target = in_cascade & (vis_flat == 1.0)         # only update untouched pixels
        vis_flat[target] = lit[target]
        visibility = vis_flat.reshape(h, w)
    return visibility


def _reconstruct_world_pos(depth: torch.Tensor, view_proj: torch.Tensor,
                           w: int, h: int) -> torch.Tensor:
    """Un-project NDC depth into world positions. Used on the CUDA path."""
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=torch.float32),
        torch.arange(w, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    ndc_x = (xx + 0.5) / w * 2.0 - 1.0
    ndc_y = 1.0 - ((yy + 0.5) / h * 2.0)
    ndc = torch.stack([ndc_x, ndc_y, depth, torch.ones_like(depth)], dim=-1)  # (H, W, 4)
    inv = torch.linalg.inv(view_proj)
    world_h = ndc.reshape(-1, 4) @ inv.T
    world = world_h[:, :3] / world_h[:, 3:4].clamp(min=1e-6)
    return world.reshape(h, w, 3)


# ------------------------------------------------------------- texture maps
_TEX_CACHE: dict[str, torch.Tensor | None] = {}


@dataclass(slots=True)
class _MaterialMaps:
    """Loaded texture tensors for a PBRMaterial (all fields optional)."""
    albedo: torch.Tensor | None = None               # (H, W, 3) linear
    normal: torch.Tensor | None = None               # (H, W, 3) tangent-space
    metallic_roughness: torch.Tensor | None = None   # (H, W, 3) glTF: G=rough, B=metal
    ao: torch.Tensor | None = None                   # (H, W, 3) R channel used

    @property
    def any(self) -> bool:
        return (self.albedo is not None or self.normal is not None
                or self.metallic_roughness is not None or self.ao is not None)

    @classmethod
    def load(cls, material) -> _MaterialMaps:        # type: ignore[no-untyped-def]
        return cls(
            albedo=_load_texture(getattr(material, "albedo_map", None), srgb=True),
            normal=_load_texture(getattr(material, "normal_map", None), srgb=False),
            metallic_roughness=_load_texture(
                getattr(material, "metallic_roughness_map", None), srgb=False),
            ao=_load_texture(getattr(material, "ao_map", None), srgb=False),
        )


def _load_texture(ref: str | None, *, srgb: bool) -> torch.Tensor | None:
    """Load a texture file into a float32 (H, W, 3) CPU tensor (cached).

    ``ref`` is a filesystem path; missing/unreadable files yield None and
    the map is silently skipped. Color maps (``srgb=True``) are decoded
    from sRGB to linear at load time.
    """
    if not ref:
        return None
    key = f"{ref}|{int(srgb)}"
    if key in _TEX_CACHE:
        return _TEX_CACHE[key]
    tex: torch.Tensor | None = None
    path = Path(ref)
    if path.is_file():
        try:
            import imageio.v3 as iio
            arr = np.asarray(iio.imread(path))
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            arr = arr[..., :3].astype(np.float32)
            if arr.max(initial=0.0) > 1.5:          # uint8/uint16 data
                arr = arr / (255.0 if arr.max() <= 255.0 else 65535.0)
            tex = torch.from_numpy(np.ascontiguousarray(arr))
            if srgb:
                tex = srgb_to_linear(tex)
        except Exception as exc:                    # noqa: BLE001
            logger.warning(f"texture load failed: {ref} ({exc})")
            tex = None
    _TEX_CACHE[key] = tex
    return tex


def _sample_uv(tex: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear texture sample at (H, W, 2) UVs; repeats at the borders."""
    h, w = tex.shape[0], tex.shape[1]
    u = uv[..., 0] % 1.0
    v = uv[..., 1] % 1.0
    px = (u * w - 0.5) % w
    py = (v * h - 0.5) % h
    x0 = px.floor().long() % w
    x1 = (x0 + 1) % w
    y0 = py.floor().long() % h
    y1 = (y0 + 1) % h
    fx = (px - px.floor()).unsqueeze(-1)
    fy = (py - py.floor()).unsqueeze(-1)
    c00 = tex[y0, x0]
    c01 = tex[y0, x1]
    c10 = tex[y1, x0]
    c11 = tex[y1, x1]
    top = c00 * (1.0 - fx) + c01 * fx
    bot = c10 * (1.0 - fx) + c11 * fx
    return top * (1.0 - fy) + bot * fy


def _apply_normal_map(nrm: torch.Tensor, tan: torch.Tensor,
                      nmap: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Tangent-space normal mapping: n' = T·nx + B·ny + N·nz."""
    ts = _sample_uv(nmap, uv) * 2.0 - 1.0
    n = nrm
    t = tan - n * (tan * n).sum(dim=-1, keepdim=True)   # re-orthogonalize
    t = t / torch.linalg.norm(t, dim=-1, keepdim=True).clamp(min=1e-9)
    b = torch.cross(n, t, dim=-1)
    out = t * ts[..., 0:1] + b * ts[..., 1:2] + n * ts[..., 2:3]
    return out / torch.linalg.norm(out, dim=-1, keepdim=True).clamp(min=1e-9)
