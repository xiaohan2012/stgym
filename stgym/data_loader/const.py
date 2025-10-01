from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetName:
    brca = "brca"
    human_crc = "human-crc"
    mouse_spleen = "mouse-spleen"
    mouse_preoptic = "mouse-preoptic"
    human_intestine = "human-intestine"
    human_lung = "human-lung"
    breast_cancer = "breast-cancer"
    mouse_kidney = "mouse-kidney"
    colorectal_cancer = "colorectal-cancer"
