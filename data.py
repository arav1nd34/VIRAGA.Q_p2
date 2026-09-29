from modules.cancer.data_pipeline import (
    load_dataset as load_cancer
)

from modules.parkinsons.data_pipeline import (
    load_dataset as load_parkinsons
)


def load_module(module_name):

    if module_name == "cancer":
        return load_cancer()

    if module_name == "parkinsons":
        return load_parkinsons()

    raise ValueError(
        f"Unknown module: {module_name}"
    )