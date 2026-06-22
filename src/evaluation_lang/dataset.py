from langsmith import Client
from src.utils import utils_helper
from src.utils.logger import eval_event


def creating_dataset(filename: str, dataset_name: str):

    client = Client()

    examples = utils_helper.load_json(filename)
    print(examples)

    # --------------------------------------------------
    # Check whether dataset already exists
    # --------------------------------------------------
    dataset = None

    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"✅ Dataset already exists: {dataset_name}")

    except Exception:

        print(f"🆕 Creating dataset: {dataset_name}")

        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Evaluation dataset for RAG pipeline"
        )
    # --------------------------------------------------
    # Check existing examples
    # --------------------------------------------------
    existing_examples = list(
        client.list_examples(
            dataset_id=dataset.id
        )
    )

    # debug
    # print(existing_examples)

    # --------------------------------------------------
    # Add examples only if dataset is empty
    # --------------------------------------------------
    if len(existing_examples) == 0:

        formatted_examples = []
        for item in examples:
            # Support two JSON shapes:
            # 1) {"question": ..., "answer": ...}
            # 2) {"inputs": {"question": ...}, "outputs": {"answer": ...}}
            if isinstance(item, dict):
                if "inputs" in item and "outputs" in item:
                    q = item["inputs"].get("question")
                    a = item["outputs"].get("answer")
                else:
                    q = item.get("question")
                    a = item.get("answer")

                if q is None or a is None:
                    # skip malformed example
                    print(f"Skipping malformed example: {item}")
                    continue

                formatted_examples.append({
                    "inputs": {"question": q},
                    "outputs": {"answer": a},
                })
            else:
                print(f"Skipping non-dict example: {item}")

        client.create_examples(
            dataset_id=dataset.id,
            examples=formatted_examples
        )

        example_count = len(formatted_examples)

        print(
            f"✅ Added {example_count} examples to dataset"
        )

    else:

        example_count = len(existing_examples)

        print(
            f"✅ Dataset already contains "
            f"{example_count} examples"
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    msg = f"""
Dataset Name : {dataset.name}
Dataset ID   : {dataset.id}
Example Count: {example_count}
"""

    eval_event("dataset.summary", dataset_name=dataset.name, dataset_id=str(dataset.id), example_count=example_count)
    print(msg)

    return {
        "dataset_name": dataset.name,
        "dataset_id": str(dataset.id),
        "example_count": example_count
    }