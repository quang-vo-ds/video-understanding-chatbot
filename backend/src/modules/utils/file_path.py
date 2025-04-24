from pathlib import Path
import os

def extract_file_base_name(file_path: str) -> str:
    return Path(file_path).stem

def extract_file_extension(file_path: str) -> str:
    return Path(file_path).suffix

def create_parent_folder(file_path: str) -> None:
    if extract_file_extension(file_path):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(file_path).mkdir(parents=True, exist_ok=True)

def get_project_dir() -> str:
    return Path(__file__).parents[4]