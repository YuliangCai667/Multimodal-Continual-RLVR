#!/usr/bin/env python3
from src.ctir.probe_dataset import freeze_navigation_probes


if __name__ == "__main__":
    freeze_navigation_probes(
        "/home/caiyuliang/datasets/MRCL/Navigation/jsons/train/data.json",
        "/home/caiyuliang/datasets/MRCL/Navigation/images",
        "experiments/ctir_t4_60/probes/navigation_probes.json",
        count=32,
        seed=142,
    )
