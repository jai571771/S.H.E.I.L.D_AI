import logging
import sys
from pathlib import Path

# Setup project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_yaml(text: str) -> dict:
    """Parses a basic indentation-based YAML string into a Python dictionary."""
    lines = text.splitlines()
    result = {}
    stack = [(-1, result)]  # stack of (indent, dict_object)
    
    for line in lines:
        # Strip comments
        line = line.split('#')[0]
        if not line.strip():
            continue
        
        indent = len(line) - len(line.lstrip(' '))
        stripped = line.strip()
        
        if ':' in stripped:
            key, val = stripped.split(':', 1)
            key = key.strip()
            val = val.strip()
        else:
            key = stripped.rstrip(':').strip()
            val = ''
            
        # Clean value
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        elif val.startswith("'") and val.endswith("'"):
            val = val[1:-1]
            
        if val.lower() == 'true':
            val = True
        elif val.lower() == 'false':
            val = False
        elif val.lower() in ('none', 'null'):
            val = None
        elif val == '':
            val = {}
        else:
            try:
                if '.' in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                pass
                
        # Pop stack until we find the parent dictionary
        while stack and stack[-1][0] >= indent:
            stack.pop()
            
        if not stack:
            # Fallback to root if stack gets empty (should not happen for valid yaml)
            stack.append((-1, result))
            
        parent_dict = stack[-1][1]
        parent_dict[key] = val
        
        if isinstance(val, dict):
            stack.append((indent, val))
            
    return result

def get_config() -> dict:
    """Loads configuration from configs/config.yaml."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    return parse_yaml(text)

def setup_logger(name: str) -> logging.Logger:
    """Sets up a standardized logger that writes to both stdout and a log file."""
    config = get_config()
    reports_dir = PROJECT_ROOT / config["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = PROJECT_ROOT / "preprocessing_pipeline.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if logger is already configured
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
        
        # Stream handler for console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
