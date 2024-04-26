from dataclasses import dataclass, field


@dataclass
class LM_Config:
    model_path: str = "mistralai/Mistral-7B-v0.1"
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    bits: int = 16
    bf16: bool = True
    fp16: bool = False

    # @classmethod
    # def from_dict(cls, data):
    #     return cls(
    #         model_path=data.get("model_path"),
    #         lora_enable=data.get(
    #             "lora_enable"
    #         ),
    #         lora_alpha = data.get("lora_alpha"),
    #         lora_r = data.get("lora_r"),
    #         lora_alpha = data.get("lora_alpha"),
    #         lora_alpha = data.get("lora_alpha"),
    #         lora_alpha = data.get("lora_alpha"),
    #     )
