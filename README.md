# NLP text classifictation

**Post-competition participation of [SEMEVAL 2022](https://github.com/Perez-AlmendrosC/dontpatronizeme) for ICL NLP.**

## Setting up the environment

To install the dependencies you can simply use

```python
python -m pip install -r requirements.txt
```

## Task

> In this task, we invite participants to detect patronizing and condescending language (PCL) in paragraphs extracted from news articles in English. Given a paragraph, systems must predict whether it contains condescending language or not (Subtask 1), and whether it contains any of the 7 categories identified in the PCL taxonomy introduced in [Perez-Almendros et al. (2020)](https://aclanthology.org/2020.coling-main.518/) (Subtask 2).

## Methods used

- Pre-trained RoBERTa for sequence classification
- Data balancing through random sampling by class weights
- Data augmentation
- Multiple losses
- Bag of Words classifier for baseline comparisons
