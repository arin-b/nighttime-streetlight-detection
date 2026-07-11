from rbccps_od.models.yolo_ablation import (
    build_yolo26_ablation_model,
    replace_c2f_with_cse as _replace_c2f_with_cse,
)


def replace_c2f_with_cse(model):
    return _replace_c2f_with_cse(getattr(model, "model", model))


def build_cse_model(weights_path="yolo26m.pt"):
    return build_yolo26_ablation_model(weights_path, use_cse=True)


def train():

    model = build_cse_model()

    model.train(
        data="configs/original.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        patience=20,
        workers=8,
        cache=True,
        project="outputs/cse_ablation",
        name="original_images"
    )


if __name__ == "__main__":
    train()
