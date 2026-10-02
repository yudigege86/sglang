# Linear-context DFlash (fork branch `dflash-linear-v0.5.19`)

Serving for `DFlashLinearDraftModel` on algorithm `DFLASH`. Stock
`DFlash2DraftModel` is unmodified v0.5.19 and stays on `DFlashWorkerV2`.

Docker images are **not** in this repo. They live in SpecForge
(`yudigege86/SpecForge` branch `dflash-linear`):

| Job | SpecForge Dockerfile | Image tag |
|---|---|---|
| **Training** | `scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.18-train` | `naqin/primus-specforge:v0.5.18-train-rocm700-mi35x` |
| **Testing / live SGLang** | `scripts/cluster/dflash-linear/Dockerfile.sglang-0.5.19` | `naqin/primus-specforge:v0.5.19-dflash-linear-rocm700-mi35x` |

- Train is SGLang **0.5.18** + FLA. It cannot serve `DFlash2DraftModel`.
- Eval/serve is SGLang **0.5.19** with this branch overlaid. `SGLANG_SRC`
  must stay on `dflash-linear-v0.5.19` (do not mount the 0.5.18
  `dflash-linear` branch onto the 0.5.19 image).
- SpecForge is bind-mounted; it is not baked into either image.

Full chooser, rebuild commands, and retired tags:
https://github.com/yudigege86/SpecForge/blob/dflash-linear/docs/linear-context-dflash-docker.md
