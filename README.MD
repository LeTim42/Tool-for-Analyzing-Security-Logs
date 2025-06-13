# Tool for Analyzing Security Logs
## About
This is a simple security log analysis tool made as a project for the Software Engineering major.

## Installation
- Install [Python](https://www.python.org/)
- Install [Ollama](https://ollama.com/)
- Install libraries `pip install -r requirements.txt`
- Download a large language model with Ollama
  - Search for models [here](https://ollama.com/search/) 
  - Example: `ollama pull llama3:8b-instruct-q8_0`
- Add queries to `queries.json` for other types of threats if needed

## Usage
- Prepare the configuration in `config.json`:
  - `logs`: Logs to analyze
    - `path`: Path to folder with logs
    - `threats`: Threats to find in logs
    - `max_lines_per_query`: Max number of lines to find for each type of threat
  - `model`: Model for interpret logs
  - `context_size`: Model's context size in tokens
- Run `main.py`
