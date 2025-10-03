from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetName:
    brca = "brca"
    brca_ptnm_m = "brca-ptnm-m"
    human_crc = "human-crc"
    mouse_spleen = "mouse-spleen"
    mouse_preoptic = "mouse-preoptic"
    human_intestine = "human-intestine"
    human_lung = "human-lung"
    breast_cancer = "breast-cancer"
    mouse_kidney = "mouse-kidney"
    cellcontrast_breast = "cellcontrast-breast"
    colorectal_cancer = "colorectal-cancer"
    upmc = "upmc"
    charville = "charville"
    brca_grade = "brca-grade"
