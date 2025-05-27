
from datasets import load_dataset


def get_dataset(dataset_name: str, chunk: int, total_chunks: int):
    """
    Loads and optionally slices a dataset by chunk index.

    Args:
        dataset_name (str): The dataset name.
        chunk (int): Which chunk index to use (-1 means full dataset).
        total_chunks (int): Total number of chunks.

    Returns:
        dataset: a Hugging Face Dataset slice.
        q_key (str): question key in the dataset.
        a_key (str): answer key in the dataset.
    """
    if dataset_name == "math500train":
        dataset = load_dataset("hendrycks/competition_math", split="train")
        assert len(dataset) == 7500
        dataset = dataset.select(list(range(0, len(dataset), 15)))
        q_key = "problem"
        a_key = "solution"
    elif dataset_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        q_key = "problem"
        a_key = "solution"
    elif dataset_name == "aime2024":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        q_key = "problem"
        a_key = "answer"
    elif dataset_name in {"aime2025", "aime2025-1"}:
        dataset = load_dataset("opencompass/AIME2025", 'AIME2025-I', split="test")
        q_key = "question"
        a_key = "answer"
    elif dataset_name == "aime2025-2":
        dataset = load_dataset("opencompass/AIME2025", 'AIME2025-II', split="test")
        q_key = "question"
        a_key = "answer"
    else:
        raise NotImplementedError(f"The dataset {dataset_name} is not implemented.")

    if chunk >= 0 and total_chunks > 1:
        chunk_size = len(dataset) // total_chunks
        start_idx = chunk * chunk_size
        end_idx = (
            len(dataset) if (chunk == total_chunks - 1)
            else (chunk + 1) * chunk_size
        )
        dataset = dataset.select(range(start_idx, end_idx))

    return dataset, q_key, a_key
