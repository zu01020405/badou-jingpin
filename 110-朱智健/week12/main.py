#-*-coding:utf-8-*-
'''
Author: Shiyao Ma
Date: 2023-10-27 22:51:57
LastEditors: Shiyao Ma
LastEditTime: 2023-10-28 00:34:10
Copyright (c) 2023 by Shiyao Ma, All Rights Reserved. 
'''
import torch
import os
import random
import os
import numpy as np
import logging
import torch.nn as nn
from config import Config
from model import choose_optimizer, TorchModel
from evaluate import Evaluator
from loader import load_data
from transformers import AutoModelForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig, \
    prepare_model_for_int8_training

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    # model = AutoModelForSequenceClassification.from_pretrained(Config["pretrain_model_path"], num_labels = Config['class_num'])
    model = AutoModelForTokenClassification.from_pretrained(Config["pretrain_model_path"], num_labels = Config['class_num'])
    # model = TorchModel(config)


    #大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)

    model = get_peft_model(model, peft_config)


    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    # cuda_flat = False
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            output = model(input_id)[0]
            loss = nn.CrossEntropyLoss()(output, torch.nn.functional.one_hot(labels).float())
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)
