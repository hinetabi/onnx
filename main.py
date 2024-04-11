from sentence_transformers import SentenceTransformer
from typing import Callable, TypeVar, Any, Union, List
from torch import Tensor
from numpy import ndarray
from codetiming import Timer
from txtai.pipeline import HFOnnx, Labels
from transformers import BertweetTokenizer, RobertaForSequenceClassification
import torch
import torch.nn.functional as F
from optimum.onnxruntime import ORTModelForSequenceClassification


_F = TypeVar("_F", bound=Callable[..., Any])
# https://stackoverflow.com/questions/653368/how-to-create-a-decorator-that-can-be-used-either-with-or-without-parameters
def timing(*args: _F, **kwargs: Any) -> Union[_F, Any]:
    """
    from ailab import timing

    @timing
    def my_func():
        sleep(30)

    # or
    @timing(text="foo")
    def my_func():
        sleep(30)
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # called as @timing
        return Timer()(args[0])
    else:
        # called as @timing("foo")
        return Timer(*args, **kwargs) # type: ignore

@timing
def sentence_sentiment_huggingface(sentences: list[str], model: Any, tokenizer: BertweetTokenizer) -> Union[List[Tensor], ndarray, Tensor]:
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, return_tensors='pt', return_token_type_ids=False)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        predicted_class_id = model_output.logits.argmax(axis=-1).tolist()

    return [model.config.id2label[id] for id in predicted_class_id]

if __name__ == "__main__":
    
    #  Sentences we want sentence embeddings for
    sentences = ['This is an example sentence'] * 20

    # Load model from HuggingFace Hub
    tokenizer = BertweetTokenizer.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')
    ort_model = ORTModelForSequenceClassification.from_pretrained('onnx/', local_files_only=True)
    # ort_model.save_pretrained("onnx/")

    sentence_sentiment = sentence_sentiment_huggingface(sentences, ort_model, tokenizer) 
    print(sentence_sentiment)
    
    
