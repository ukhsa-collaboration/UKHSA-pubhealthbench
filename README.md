# PubHealthBench

This is a fork of the [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro/tree/main) evaluation code, refactored for running PubHealthBench.

| [**📖Paper**](https://arxiv.org/abs/2505.06046) | [**🤗 Dataset**](https://huggingface.co/datasets/Joshua-Harris/PubHealthBench) |

## Introduction

PubHealthBench is a benchmark designed to provide a broad assessment of LLM knowledge of current UK Government public health guidance. PubHealthBench contains over 8000 questions relating to public, clinical, and professional guidance across 10 public health topic areas, sourced from 687 documents from the UK Government website (gov.uk) on 08/01/2025.

## Dataset Creation
We use a large corpus of 1,150 current UK Government guidance documents from the UK Government website (gov.uk) in HTML and PDF formats as the source material. To generate PubHealthBench we develop an automated pipeline to extract free text from documents, chunk it into sections, generate MCQA samples, and filter to a high quality subset. See the paper for full details.

## API Evaluation

To use the API for inference, set your API_KEY as an environment variable then run:

**MCQA Setup:**

```bash
python evaluate_from_api.py \  
                 --model_name gpt-4o-mini \
                 --output_dir eval_results_reviewed_mcqa \
                 --assigned_subjects all \
                 --subset reviewed \
                 --setup mcqa

```

**Free Form Setup:**

To use the API for inference, set your API_KEY and JUDGE_API_KEY as environment variables then run:

```bash
 python evaluate_from_api.py \  
                 --model_name gpt-4o-mini \     
                 --output_dir eval_results_reviewed_freeform \
                 --assigned_subjects all \
                 --subset reviewed \
                 --setup freeform \
                 --judge_model_name gpt-4o-mini-2024-07-18
```

### Overall Accuracy
```
python compute_accuracy.py eval_results_reviewed_mcqa/
```

## Local Evaluation

To use local inference, ensure you have vLLM installed and run:

**MCQA Setup:**

```bash
python evaluate_from_local.py \  
                 --model google/gemma-3-1b-it \
                 --save_dir eval_results_reviewed_mcqa \
                 --selected_subjects all \
                 --subset=reviewed 
```

Note we do not currently provide a fully local implementation of the free form setup as we only evaluate gpt-4o-mini-2024-07-18 as a judge.


## Citation

**BibTeX:**

PubHealthBench
```bibtex
@misc{harris2025healthyllmsbenchmarkingllm,
      title={Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information}, 
      author={Joshua Harris and Fan Grayson and Felix Feldman and Timothy Laurence and Toby Nonnenmacher and Oliver Higgins and Leo Loman and Selina Patel and Thomas Finnie and Samuel Collins and Michael Borowitz},
      year={2025},
      eprint={2505.06046},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.06046}, 
}
```

MMLU-Pro (original evaluation code in this repo)
```bibtex
@article{wang2024mmlu,
  title={Mmlu-pro: A more robust and challenging multi-task language understanding benchmark},
  author={Wang, Yubo and Ma, Xueguang and Zhang, Ge and Ni, Yuansheng and Chandra, Abhranil and Guo, Shiguang and Ren, Weiming and Arulraj, Aaran and He, Xuan and Jiang, Ziyan and others},
  journal={arXiv preprint arXiv:2406.01574},
  year={2024}
}
```
