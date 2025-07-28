import torch

from typing import List
from rouge import Rouge
from sklearn.metrics import f1_score

from evaluation import qa_utils
from logger_config import logger
import nltk
nltk.data.path.append('/public/home/zhangzheng2024/nltk_data')
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import re, string
# from datasets import load_metric #老版本，现在不再使用

@torch.no_grad()
def accuracy(output: torch.tensor, target: torch.tensor, topk=(1,)) -> List[float]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk) #1
    batch_size = target.size(0) #16

    _, pred = output.topk(maxk, 1, True, True)  #pred：16*1 每个样本前maxk(1)个预测值的索引
    pred = pred.t()#1*16 把预测的索引转为1行
    correct = pred.eq(target.view(1, -1).expand_as(pred))#一行true false 表示pred和target是否相同

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True) #正确的个数
        res.append(correct_k.mul_(100.0 / batch_size).item())#一个batch中的正确数，除以batch size
    return res


@torch.no_grad()
def batch_mrr(output: torch.tensor, target: torch.tensor) -> float:
    assert len(output.shape) == 2
    assert len(target.shape) == 1
    sorted_score, sorted_indices = torch.sort(output, dim=-1, descending=True)
    _, rank = torch.nonzero(sorted_indices.eq(target.unsqueeze(-1)).long(), as_tuple=True)
    assert rank.shape[0] == output.shape[0]

    rank = rank + 1
    mrr = torch.sum(100 / rank.float()) / rank.shape[0]
    return mrr.item()


## =========================================================================== ##

# Copy from https://github.com/microsoft/LMOps/tree/main/uprise/src/utils


def rouge(preds, labels):
    # https://github.com/pltrdy/rouge
    r1s, r2s, rls = [], [], []
    r = Rouge()
    for i in range(len(labels)):
        if '\n' not in preds[i]: preds[i] += '\n'
        if '\n' not in labels[i]: labels[i] += '\n'  # avoid empty string
        scores = r.get_scores(preds[i], labels[i])[0]
        r1s.append(scores["rouge-1"]['f'])
        r2s.append(scores["rouge-2"]['f'])
        rls.append(scores["rouge-l"]['f'])
    r1 = sum(r1s) / len(r1s)
    r2 = sum(r2s) / len(r2s)
    rl = sum(rls) / len(rls)
    return r1, r2, rl


def squad(labels, preds):
    """Computes SQuAD metrics, maximizing over answers per question.
    Args:
    labels: list of lists of strings
    preds: list of strings
    Returns:
    dict with score_key: squad score across all labels and predictions
    """
    labels = [[qa_utils.normalize_squad(t) for t in u] for u in labels]
    preds = [qa_utils.normalize_squad(p) for p in preds]
    em, f1 = qa_utils.qa_metrics(labels, preds)  # em,f1
    return em, f1


def trivia_qa(labels, preds):
    """Computes TriviaQA metrics, maximizing over answers per question.
    Args:
    labels: list of lists of strings
    preds: list of strings
    Returns:
    dict with score_key: squad score across all labels and preds
    """
    labels = [[qa_utils.normalize_trivia_qa(t) for t in u] for u in labels]
    preds = [qa_utils.normalize_trivia_qa(p) for p in preds]
    em, f1 = qa_utils.qa_metrics(labels, preds)  # em,f1
    return em, f1


def simple_accuracy(preds, labels):
    if isinstance(preds[0], str):
        labels = [label.lower().strip() for label in labels]
        preds = [pred.lower().strip() for pred in preds]
    res = [int(preds[i] == labels[i]) for i in range(len(preds))]
    acc = sum(res) / len(res)
    return acc


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    # Currently only MRPC & QQP use this metric
    f1 = f1_score(y_true=labels, y_pred=preds, pos_label='Yes')
    return acc, f1, (acc + f1) / 2

def compute_sentence_bleu_scores(preds, labels):
    """    
    计算 BLEU-1、BLEU-3、BLEU-4 的平均分。
    preds: list of strings
    labels: list of strings
    
    我们将分别计算每对（pred, label）之间的BLEU分数，然后求平均。
    我们假设输入的labels与preds是一一对应的单句评估。
    """
    smoothie = SmoothingFunction().method1
    
    # 对字符串进行简单分词，这里假设空格分词
    # 更严格的情况可根据具体需要进行tokenize
    def tokenize(s):
        return s.split()
    
    bleu1_scores = []
    bleu3_scores = []
    bleu4_scores = []
    
    for p, l in zip(preds, labels):
        ref = [tokenize(l)]  # reference是二维列表
        hyp = tokenize(p)
        
        # BLEU-1: weights=(1,0,0,0)
        bleu1 = sentence_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu1_scores.append(bleu1)
        
        # BLEU-3: weights=(1/3,1/3,1/3,0)
        bleu3 = sentence_bleu(ref, hyp, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie)
        bleu3_scores.append(bleu3)
        
        # BLEU-4: weights=(0.25,0.25,0.25,0.25)
        bleu4 = sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        bleu4_scores.append(bleu4)
    
    return (sum(bleu1_scores)/len(bleu1_scores)*100,
            sum(bleu3_scores)/len(bleu3_scores)*100,
            sum(bleu4_scores)/len(bleu4_scores)*100)


def compute_corpus_bleu_scores(preds, labels):
    """
    使用corpus_bleu计算BLEU-1、BLEU-3、BLEU-4。
    """
    punctuation = '[%s]+' % re.escape(string.punctuation)

    ref_list = []
    hyp_list = []

    for ref, hyp in zip(labels, preds):
        # 去除标点并转小写（根据对方代码逻辑）
        ref = ref.strip().lower()
        hyp = hyp.strip().lower()
        ref = re.sub(punctuation, '', ref)
        hyp = re.sub(punctuation, '', hyp)

        # 分词
        ref_tokens = word_tokenize(ref)
        hyp_tokens = word_tokenize(hyp)

        # corpus_bleu的参考结构要求为：list of list of list
        # 每个预测的参考需要嵌套两层list，因此这里为[[ref_tokens]]
        ref_list.append([ref_tokens])
        hyp_list.append(hyp_tokens)

    # BLEU-1
    bleu1 = corpus_bleu(ref_list, hyp_list, weights=(1, 0, 0, 0)) * 100
    # BLEU-3: weights=(1/3,1/3,1/3,0)
    bleu3 = corpus_bleu(ref_list, hyp_list, weights=(1/3, 1/3, 1/3, 0)) * 100
    # BLEU-4: 默认corpus_bleu不传weights即为BLEU-4或使用(0.25,0.25,0.25,0.25)
    bleu4 = corpus_bleu(ref_list, hyp_list) * 100

    return bleu1, bleu3, bleu4


def compute_bleu(preds, labels):
    from evaluate import load 
    bleu = load("bleu", cache_dir="tools/evaluate_cache")
    predictions = preds  # preds为一个字符串列表，如 ["This is a test", "Another test"]
    references = labels
    # with open("log/predictions_new.txt", "w", encoding="utf-8") as pred_file:
    #     for pred in predictions:
    #         pred_file.write(pred)
    # with open("log/references_new.txt", "w", encoding="utf-8") as ref_file:
    #     for ref in references:
    #         ref_file.write(ref)
    return bleu.compute(predictions=predictions, references=references)

def compute_metrics(metric, labels, preds):
    assert len(preds) == len(labels)
    if metric == 'simple_accuracy':
        return {'acc': simple_accuracy(preds, labels) * 100}
    elif metric == 'rouge':
        r1, r2, rl = rouge(preds, labels)
        return {'r1': r1 * 100, 'r2': r2 * 100, 'rl': rl * 100}
    elif metric == 'acc_and_f1':
        acc, f1, acc_f1 = acc_and_f1(preds, labels)
        return {'acc': acc * 100, 'f1': f1 * 100, 'acc_and_f1': acc_f1 * 100}
    elif metric == 'f1':
        f1 = f1_score(y_true=labels, y_pred=preds, pos_label='Yes')
        return {'f1': f1 * 100}
    elif metric == 'squad':
        em, f1 = squad(labels=labels, preds=preds)
        return {'em': em, 'f1': f1}
    elif metric == 'trivia_qa':
        em, f1 = trivia_qa(labels=labels, preds=preds)
        return {'em': em, 'f1': f1}
    elif metric == 'generation':
        # sentence_bleu1, sentence_bleu3, sentence_bleu4 = compute_sentence_bleu_scores(preds, labels)
        bleu1, bleu3, bleu4 = compute_corpus_bleu_scores(preds, labels)
        em = simple_accuracy(preds, labels)*100
        r1, r2, rl = rouge(preds, labels)
        bleu = compute_bleu(preds=preds, labels=labels) #align with MDR
        return {
            # 'sentence_bleu1': sentence_bleu1,
            # 'sentence_bleu3': sentence_bleu3,
            # 'sentence_bleu4': sentence_bleu4,
            'UDR_bleu1': bleu1,
            'UDR_bleu3': bleu3,
            'UDR_bleu4': bleu4,
            "bleu": bleu["bleu"] * 100,
            'em': em,
            'r1': r1 * 100,
            'r2': r2 * 100,
            'rl': rl * 100
        }
    else:
        raise ValueError(metric)
