# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.chatglm import TransformerChatGlmLMHeadModel, setup_model_profile, ChatGLMConfig, \
    ChatGLMForConditionalGeneration
from deep_training.nlp.models.lora import LoraArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config
from tokenization_chatglm import ChatGLMTokenizer


class MyTransformer(TransformerChatGlmLMHeadModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, _ = parser.parse_dict(train_info_args, allow_extra_keys=True)

    setup_model_profile()

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer: ChatGLMTokenizer
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=ChatGLMTokenizer, config_class_name=ChatGLMConfig)

    # 官方28层
    config.num_layers = 28
    model = MyTransformer(config=config, model_args=model_args, training_args=training_args)

    model.eval()

    base_model: ChatGLMForConditionalGeneration = model.backbone.model
    # 按需修改，目前只支持 4/8 bit 量化
    base_model.half().to(torch.device('cuda:3'))

    with torch.inference_mode():
        prefixs = [
            "我想听一首开心的歌曲",
            "周五下班了但工作没做完，不太开心",
            "我想听听一首风格的西方通俗歌曲，希望它是国语，我希望它是原唱。"
            "我想听听一首风格的西方通俗歌曲，希望它是中国话，它应该是一首演绎水平的歌曲，我希望听到爱情的感觉，我希望它是抖音。"
            "上山打老虎的人应该听什么歌？"
            "谈恋爱了，我应该听什么歌？"
        ]
        for question in prefixs:
            response, history = base_model.chat(tokenizer, question, history=[], max_length=1024)
            print(question, ' ', response)

        # response, history = base_model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=30)
        # print('写一个诗歌，关于冬天',' ',response)
