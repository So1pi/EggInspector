import json
from bs4 import BeautifulSoup


def parse_config(config_file):
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        print(f"Config 파일 '{config_file}'을 찾을 수 없습니다.")
        return None

