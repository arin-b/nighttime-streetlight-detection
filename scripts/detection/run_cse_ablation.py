from rbccps_od.training.train_yolo26m_cse import build_cse_model


EXPERIMENTS = [

    {
        "name": "original",
        "data": "src/rbccps_od/config/original_images.yaml"
    },

    {
        "name": "zerodce",
        "data": "src/rbccps_od/config/zerodce.yaml"
    },

    {
        "name": "retinex",
        "data": "src/rbccps_od/config/retinex.yaml"
    },

    {
        "name": "retinex_zerodce",
        "data": "src/rbccps_od/config/retinex_zerodce.yaml"
    }
]


def main():
    for exp in EXPERIMENTS:

        model = build_cse_model()

        model.train(
            data=exp["data"],
            epochs=100,
            imgsz=640,
            batch=16,
            device=0,
            optimizer="AdamW",
            lr0=1e-3,
            project="outputs/cse_ablation",
            name=exp["name"]
        )


if __name__ == "__main__":
    main()
