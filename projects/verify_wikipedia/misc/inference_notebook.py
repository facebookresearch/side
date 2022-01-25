import hydra
from hydra import compose, initialize

from omegaconf import OmegaConf

from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state
from dpr.utils.dist_utils import setup_cfg_gpu
from dpr.utils.model_utils import load_states_from_checkpoint, get_model_obj

def get_crossencoder_components_from_checkpoint(cfg):

    initialize(config_path="../conf/")
    cfg = compose(
        config_name="retrieval",
        overrides=cfg,
    )

    # print(OmegaConf.to_yaml(cfg))

    # Load all the configs
    checkpoint_config = OmegaConf.load(f"{cfg.retriever.checkpoint_dir}/.hydra/config.yaml")
    checkpoint_cfg = setup_cfg_gpu(checkpoint_config)

    # Load weights
    saved_state = load_states_from_checkpoint(f"{cfg.retriever.model_file}")
    set_cfg_params_from_state(saved_state.encoder_params, checkpoint_cfg)

    # Get model and tensorizer
    tensorizer, biencoder, _ = init_biencoder_components(
        checkpoint_cfg.base_model.encoder_model_type, checkpoint_cfg, inference_only=True
    )

    # Set model to eval
    biencoder.eval()

    # load weights from the model file
    biencoder = get_model_obj(biencoder)
    biencoder.load_state(saved_state, strict=True)

    # Instantiate dataset
    ctx_src = hydra.utils.instantiate(cfg.retriever.datasets[cfg.retriever.ctx_src], checkpoint_cfg, tensorizer)

    # Get helper functions
    loss_func = biencoder.get_loss_function()
    biencoder_prepare_encoder_inputs_func = biencoder.prepare_model_inputs

    return ctx_src, biencoder_prepare_encoder_inputs_func, biencoder