
from langsmith import Client
import json
from pathlib import Path
import os



def load_json(filename: str) -> dict:
    current_dir = Path(__file__).resolve().parent.parent
    # print(current_dir)

    json_path=os.path.join(current_dir, "evaluation_lang",filename)
    print(json_path)


    try:
    # 3. Read and parse the file safely
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file) # Returns a Python dictionary

    except FileNotFoundError:
        print(f"Error: The file at {json_path} could not be found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file at {json_path} contains invalid JSON syntax.")
        return {}
