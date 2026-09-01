def get_compass_vars(architecture_template_path: str | None):
    if not architecture_template_path:
        return None
    from LLMCompass.design_space_exploration.dse import (
        read_architecture_template,
        template_to_system,
    )
    from LLMCompass.software_model.transformer import (
        TransformerBlockAutoRegressionTP,
        TransformerBlockInitComputationTP,
    )
    from LLMCompass.software_model.utils import data_type_dict

    specs = read_architecture_template(architecture_template_path)
    system = template_to_system(specs)
    prefill_model = TransformerBlockInitComputationTP(
        d_model=4096,
        n_heads=32,
        device_count=1,
        data_type=data_type_dict["fp16"],
    )
    decode_model = TransformerBlockAutoRegressionTP(
        d_model=4096,
        n_heads=32,
        device_count=1,
        data_type=data_type_dict["fp16"],
    )
    wrapped_llmcompass_vars = (system, prefill_model, decode_model)
    return wrapped_llmcompass_vars
