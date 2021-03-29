


def run_model(models, features):
    # First decision
    for experiment in  ["Invasive v.s. Noninvasive",
                        "Atypia and DCIS v.s. Benign",
                        "DCIS v.s. Atypia"]:
        pca = models[experiment + " PCA"]
        if pca is not None:
            features = pca.transform(features).reshape(1, -1)
        model = models[experiment + " model"]
        rst = model.predict(features)[0]
    
        if rst:
            if experiment == "Invasive v.s. Noninvasive":
                return 4, "Invasive"
            if experiment == "Atypia and DCIS v.s. Benign":
                return 1, "Benign"
            if experiment == "DCIS v.s. Atypia":
                return 3, "DCIS"
            raise("programming error! unknown experiment")

    if experiment == "DCIS v.s. Atypia" and not rst:
        return 2, "Atypia"
    
    raise("programming error 2! Unknown experiment and rst")
    
    
    