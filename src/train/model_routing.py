"""Model-to-training-step routing helpers."""

CLASSIFIER_MODELS = {
    "classifier",
    "classifier_stm",
    "classifier_se",
    "clf_attnpl",
    "clf_attnpl_t",
    "classifier_tm_s",
    "clf_tm_attnpl",
    "clf_tm_attnpl_t",
    "classifier_stm_s",
    "clf_tm_cv_attnpl_t",
    "clf_mixer_attnpl_t",
    "clf_rnn",
}

REGRESSOR_MODELS = {
    "regressor",
    "lstm",
    "lstm_legacy",
    "reg_attnpl",
    "reg_mixer_attnpl_t",
    "reg_rnn",
}

TPP_MODELS = {
    "thp",
    "rtpp",
    "rtpp_v2",
    "oracle",
    "mtpp",
    "thp_deltat",
    "mhp",
    "btpp",
    "etas",
    "etas_zhuang",
    "fast_netas",
    "netas",
    "netas_fast",
    "nhpp",
    "njdtpp",
}

TPP_M_MODELS = {
    "mixer_tpp",
}


def get_model_family(model_name):
    if model_name in CLASSIFIER_MODELS:
        return "classifier"
    if model_name in REGRESSOR_MODELS:
        return "regressor"
    if model_name in TPP_MODELS:
        return "tpp"
    if model_name in TPP_M_MODELS:
        return "tpp_m"
    raise ValueError(f"Unsupported model class: {model_name}")


def get_train_step_module(model_name):
    family = get_model_family(model_name)
    family_to_module = {
        "classifier": "classifier_train_step",
        "regressor": "regressor_train_step",
        "tpp": "tpp_train_step",
        "tpp_m": "tpp_m_train_step",
    }
    return family_to_module[family]


def is_tpp_family(model_name):
    return get_model_family(model_name) in {"tpp", "tpp_m"}
