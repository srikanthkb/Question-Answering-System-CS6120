#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 20:57:31 2022

@author: premkumar
"""

import collections
import torch

from transformers import (WEIGHTS_NAME,
                          BertConfig, BertForQuestionAnswering, BertTokenizer,
                          XLMConfig, XLMForQuestionAnswering, XLMTokenizer,
                          XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer,
                          DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)

from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from utils_squad import (get_squad_example_object, convert_example_to_model_inputs, get_answer)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

MODEL_CLASSES = {
                    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
                    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
                    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
                    'distilbert': (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
                }

def to_list(tensor):
    return tensor.detach().cpu().tolist()


class Model:

    def __init__(self, model_type, model_path, tokenizer_path):
        
        self.model_type = model_type
        self.max_seq_length = 384
        self.doc_stride = 128
        self.max_query_length = 64
        self.do_lower_case = True
        self.n_best_size = 20
        self.max_answer_length = 30
        self.eval_batch_size = 1
        self.model, self.tokenizer = self.model_load(model_path, tokenizer_path)
        self.model.eval()

    def model_load(self, model_path, tokenizer_path):
        
        model_config, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]
        config = model_config.from_pretrained(model_path + "/config.json")
        tokenizer = tokenizer_class.from_pretrained(tokenizer_path, do_lower_case=self.do_lower_case)
        model = model_class.from_pretrained(model_path, config=config)
        return model, tokenizer

    def predict(self, context, question):

        example = get_squad_example_object(context, question)
        features = convert_example_to_model_inputs(
            example, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index)


        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)

        all_results = []
        for batch in (eval_dataloader):
            batch = tuple(t for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1]}
                example_indices = batch[3]
                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(
                    unique_id = unique_id,
                    start_logits = to_list(outputs[0][i]),
                    end_logits = to_list(outputs[1][i]))
                all_results.append(result)

        answer = get_answer(
            example,
            features,
            all_results,
            self.do_lower_case,
            self.n_best_size,
            self.max_answer_length)
        
        return answer
    
if __name__ == '__main__':
    
    model = Model(
        model_type = 'bert',
        model_path = './Final_Checkpoints_Bert',
        tokenizer_path = './Final_Checkpoints_Bert/tokenizer')
    
    context = "World War II (often abbreviated to WWII or WW2), also known as the Second World War, \
               was a global war that lasted from 1939 to 1945. The vast majority of the world's countries—including all \
               the great powers—eventually formed two opposing military alliances: the Allies and the Axis. A state of total war emerged, \
               directly involving more than 100 million people from more than 30 countries. The major participants threw their entire economic, \
               industrial, and scientific capabilities behind the war effort, blurring the distinction between civilian and military resources. \
               World War II was the deadliest conflict in human history, marked by 70 to 85 million fatalities, most of whom were civilians \
               in the Soviet Union and China. It included massacres, the genocide of the Holocaust, strategic bombing,\
               premeditated death from starvation and disease, and the only use of nuclear weapons in war."
    
    question = "What years did WW2 last between?"
    
    answer = model.predict(context, question)
    
    
    print("Question: " + question)
    print("Answer: " + answer["answer"])
    