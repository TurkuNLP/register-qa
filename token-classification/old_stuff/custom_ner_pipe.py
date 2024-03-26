import transformers
import numpy as np



# this is copied and modified from:
# https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py
# wee needed to get scores for each class, not only the winner, but retain the formatting of pipeline
# hence, returning all scores added here

# difference to 
# transformers.pipeline("ner", model=model_name, tokenizer=tokenizer_name, aggregation_strategy=None, ignore_labels=[""],device=0)
# ?
# none, except some error in the scores of magnitude 1e-7 SEE custom_pipe_test.ipynb


class custom_ner_pipe:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # Here is the function that should be used:
    
    def predict(self, text):
        tok = self.tokenizer(text, return_tensors='pt',return_offsets_mapping=True,truncation=True)
        input_ids = tok.input_ids[0]
        offset_mapping = tok.offset_mapping[0]
        output = self.model(**self.tokenizer(text, return_tensors='pt',truncation=True).to(self.device))
        logits = output["logits"][0].cpu().detach().numpy()  
        return self.postprocess(output, [logits], [text], [input_ids], [offset_mapping], aggregation_strategy=None, ignore_labels=None)

    def gather_pre_entities(self,
        sentence: str,
        input_ids: np.ndarray,
        scores: np.ndarray,
        offset_mapping: list,#Optional[List[Tuple[int, int]]],
        special_tokens_mask: np.ndarray,
        aggregation_strategy: str,
    ) -> list[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            # Filter special_tokens
            if special_tokens_mask[idx]:
                continue
            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                if not isinstance(start_ind, int):
                    if True: #######framework == "pt":
                        start_ind = start_ind.item()
                        end_ind = end_ind.item()
                word_ref = sentence[start_ind:end_ind]
                if getattr(self.tokenizer, "_tokenizer", None) and getattr(self.tokenizer._tokenizer.model, "continuing_subword_prefix", None):
                    # This is a BPE, word aware tokenizer, there is a correct way
                    # to fuse tokens
                    is_subword = len(word) != len(word_ref)
                else:
                    # This is a fallback heuristic. This will fail most likely on any kind of text + punctuation mixtures that will be considered "words". 
                    # Non word aware models cannot do better than this unfortunately.
                    #if aggregation_strategy in {
                    #    AggregationStrategy.FIRST,
                    #    AggregationStrategy.AVERAGE,
                    #    AggregationStrategy.MAX,
                    #}:
                    #    warnings.warn(
                    #        "Tokenizer does not support real words, using fallback heuristic",
                    #        UserWarning,
                    #    )
                    is_subword = start_ind > 0 and " " not in sentence[start_ind - 1 : start_ind + 1]
                    
    
                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                start_ind = None
                end_ind = None
                is_subword = False
    
            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities
    
    
    def aggregate_overlapping_entities(self, entities):
        if len(entities) == 0:
            return entities
        entities = sorted(entities, key=lambda x: x["start"])
        aggregated_entities = []
        previous_entity = entities[0]
        for entity in entities:
            if previous_entity["start"] <= entity["start"] < previous_entity["end"]:
                current_length = entity["end"] - entity["start"]
                previous_length = previous_entity["end"] - previous_entity["start"]
                if current_length > previous_length:
                    previous_entity = entity
                elif current_length == previous_length and entity["score"] > previous_entity["score"]:
                    previous_entity = entity
            else:
                aggregated_entities.append(previous_entity)
                previous_entity = entity
        aggregated_entities.append(previous_entity)
        return aggregated_entities
    
    def aggregate(self, pre_entities: list[dict], aggregation_strategy: str) -> list[dict]:
        if aggregation_strategy is None:
            entities = []
            for pre_entity in pre_entities:
                entity_idx = pre_entity["scores"].argmax()
                score = pre_entity["scores"][entity_idx]
                entity = {
                    "entity": self.model.config.id2label[entity_idx],
                    "score": score,
                    "full_score": pre_entity["scores"],
                    "full_score_names": self.model.config.id2label,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                entities.append(entity)
        else:
            print("NOT IMPLEMENTED")
            #entities = self.aggregate_words(pre_entities, aggregation_strategy)
    
        if aggregation_strategy == None:#AggregationStrategy.NONE:
            return entities
        #return self.group_entities(entities)
    
    def postprocess(self, all_outputs, logits_list, text_list, input_ids_list, offset_mapping_list, aggregation_strategy=None, ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = ["O"]
        all_entities = []
        for logs, text, input_id, omapp in zip(logits_list, text_list, input_ids_list, offset_mapping_list):
            logits = logs #model_outputs["logits"][0].detach().numpy()                 # DETACH ADDED
            sentence = text # all_outputs[0]["sentence"]
            input_ids = input_id #model_outputs["input_ids"][0]
            #offset_mapping = (
            #    model_outputs["offset_mapping"][0] if model_outputs["offset_mapping"] is not None else None
            #)
            offset_mapping = omapp
            special_tokens_mask = np.array([False]*len(input_ids))#model_outputs["special_tokens_mask"][0].numpy()
    
            maxes = np.max(logits, axis=-1, keepdims=True)
            shifted_exp = np.exp(logits - maxes)
            scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
    
            pre_entities = self.gather_pre_entities(
                sentence, input_ids, scores, offset_mapping, special_tokens_mask, aggregation_strategy
            )
            grouped_entities = self.aggregate(pre_entities, aggregation_strategy)
            # Filter anything that is in self.ignore_labels
            #entities = [
            #    entity
            #    for entity in grouped_entities
            #    if entity.get("entity", None) not in ignore_labels
            #    and entity.get("entity_group", None) not in ignore_labels
            #]
            entities = [entity for entity in grouped_entities if entity["word"] not in self.tokenizer.special_tokens_map.values()]
            all_entities.extend(entities)
        num_chunks = len(all_outputs)
        if num_chunks > 1:
            all_entities = self.aggregate_overlapping_entities(all_entities)
        return all_entities
    
    
    