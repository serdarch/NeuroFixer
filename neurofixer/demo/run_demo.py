"""Executable smoke demo: python -m neurofixer.demo.run_demo --backbone cnn"""
from __future__ import annotations

from .arguments import build_parser
from .functions import run_smoke_demo


def main() -> None:
    args = build_parser().parse_args()
    result = run_smoke_demo(
        backbone=args.backbone,
        num_classes=args.num_classes,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        patch_size=args.patch_size,
        device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
