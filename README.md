# LessonDiv
````markdown
# LessonDiv

LessonDiv is a project for **teaching-plan stage division** with large language models.  
It supports single-file testing, multi-file evaluation, RAG-based assistance, and classifier-based auxiliary experiments.

The main goal is:

1. identify the teaching stage names in a lesson plan  
2. divide the complete text content belonging to each stage  
3. evaluate the division quality with multiple metrics

---

## Project Structure

```text
LessonDiv/
├── Classer/
├── file procession/
├── Others_RAG/
├── RAG/
├── RAGrecycle/
├── ST-RAG/
├── ST-RAG_2/
├── ST-RAG_3/
├── test_history/
├── bgeapi.py
├── lesson_mode_cls1.pkl
├── lesson_type_cls.pkl
├── LLMtest_evaluate8_MoonAPI.py
├── LLMtest_evaluate8.py
├── LLMtest_evaluate8_DeepseekAPI.py
├── LLMtest_evaluate9.py
├── macro_template.jsonl
├── readme.txt
├── rerankapi.py
├── testrerankapi.py
├── train_classifier.py
├── trans_micro.json.py
├── trans_prompt.json.py
````

---

## Main Script Evolution

### Early versions

* **LLMtest4_gemma**
  Uses `client.completions.create` instead of chat-style interface.
  It contains 3 functions that do not use `chat`.

* **LLMtest6**
  Revised version of `LLMtest4_gemma`.
  Function definitions were removed and the script was flattened.

* **LLMtest7**
  Universal template version.

* **LLMtest9**
  Optimized version of `LLMtest7`.
  Poor-performing models were removed and better-performing models were kept.
  Used for **single lesson-plan testing**.

---

## Multi-file Lesson Division

* **LLMtest10**
  Used for dividing processed `.txt` files into teaching stages.
  It performs:

  1. stage name recognition
  2. full content assignment under each stage
     Used for **multi-file lesson-plan testing without metrics**.

* **LLMtest11**
  Enhanced prompt version of `LLMtest10`.
  The prompt is improved mainly in the `message` part.

---

## Evaluation Scripts

* **LLMtest_evaluate**
  Adds evaluation metrics based on `LLMtest11`.
  Reference answers come from `teaching_plan.json`.
  This version does **not** print each file result in the output box.

* **LLMtest_evaluate1**
  Restores per-file printing on the basis of `LLMtest_evaluate`.
  Also adds an evaluation summary to the output JSON.

* **LLMtest_evaluate5**
  Newer version for multi-file evaluation with metrics.
  Evaluation metrics are included, but still need further optimization.

* **LLMtest_evaluate6**
  Adds prompt instructions for reconstructing shuffled teaching-plan order.
  It is designed for datasets where stage names are **deleted and shuffled**.
  `evaluate5` only corresponds to datasets with **deleted stage names**.

* **LLMtest_evaluate7**
  Based on `evaluate6`, removes 4 less important metrics.
  Adds:

  * stage boundary F1
  * stage name normalization
    Final metrics include 4 items:
  * content completeness
  * order rationality
  * stage boundary F1
  * stage name normalization

* **LLMtest_evaluate8**
  Based on `evaluate7`, further improves the stage-boundary judgment rule.
  Also fixes the order-rationality scoring:

  * when predicted text or ground-truth text is empty
  * when TF-IDF vectorization fails
  * when there is only one stage
    the old default `0.5` is changed to `0` for more reasonable scoring.

  This is the main version for **multi-file lesson-plan evaluation with metrics**, especially for:

  * deleted stage names
  * shuffled stage order

---

## Recommended Experiment Settings

### Use `LLMtest_evaluate8.py` by default for:

* Deepseek-R1-1.5B
* Qwen3-4B
* internlm2_5-7b-chat-1m
* MiniCPM3-4B

### Use `LLMtest_evaluate9.py` for:

* MiniCPM3-4B
* Deepseek7b

This corresponds to using the **opposite default setting**.

### Use `LLMtest_evaluate10.py` for:

* Qwen1.5-4B-chat
* Chinese-Mistral-7B-Instruct-v0.1
* chatglm3-6b
* Baichuan2-7B-Chat
* internlm2_5-7b-chat-1m

---

## Supporting Files

* **`macro_template.jsonl`**
  Prompt template or macro configuration file.

* **`lesson_mode_cls1.pkl` / `lesson_type_cls.pkl`**
  Saved classifier models.

* **`train_classifier.py`**
  Script for classifier training.

* **`bgeapi.py`**
  Embedding API-related script.

* **`rerankapi.py` / `testrerankapi.py`**
  Reranking API scripts and test scripts.

* **`trans_micro.json.py` / `trans_prompt.json.py`**
  Data or prompt conversion scripts.

* **`Classer/`**
  Classifier-related resources.

* **`RAG/`, `Others_RAG/`, `RAGrecycle/`, `ST-RAG/`, `ST-RAG_2/`, `ST-RAG_3/`**
  Retrieval-augmented generation experiments and different RAG variants.

* **`file procession/`**
  File preprocessing scripts.

* **`test_history/`**
  Historical experiment results or test records.

---

## Typical Workflow

### 1. Single lesson-plan testing

Use:

* `LLMtest7`
* `LLMtest9`

### 2. Multi-file stage division without metrics

Use:

* `LLMtest10`
* `LLMtest11`

### 3. Multi-file stage division with metrics

Prefer:

* `LLMtest_evaluate8`

### 4. Model-specific experiments

Use:

* `LLMtest_evaluate8`
* `LLMtest_evaluate9`
* `LLMtest_evaluate10`

according to the recommended model settings above.

---

## Evaluation Focus

The later versions mainly focus on the following metrics:

* **Content Completeness**
* **Order Rationality**
* **Stage Boundary F1**
* **Stage Name Normalization**

Among them, `LLMtest_evaluate8` is currently the preferred version because it improves:

* boundary judgment
* empty-text handling
* TF-IDF exception handling
* single-stage scoring behavior

---

## Suggested Usage

If you are starting experiments, the recommended order is:

1. preprocess lesson-plan text
2. run `LLMtest10` / `LLMtest11` for stage division
3. run `LLMtest_evaluate8.py` for evaluation
4. switch to `evaluate9` or `evaluate10` only for specific model settings

---

## Notes

* `LLMtest_evaluate8.py` is the default recommended evaluation script.
* `LLMtest9.py` is mainly for **single lesson-plan testing**.
* `LLMtest10.py` and `LLMtest11.py` are for **multi-file lesson-plan division**.
* `LLMtest_evaluate5` and later versions are for **multi-file evaluation with metrics**.
* For new experiments, prioritize the latest stable evaluation version instead of older scripts.

---

## Quick Recommendation

* **Single lesson-plan test**: `LLMtest9.py`
* **Multi-file division without metrics**: `LLMtest11.py`
* **Multi-file division with metrics**: `LLMtest_evaluate8.py`

```
