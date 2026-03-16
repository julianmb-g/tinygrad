import pytest
import sys
import os

def should_ignore(path):
    if not path.endswith(".py"): return False
    try:
        with open(path, "r") as f:
            content = f.read()
            if "import torch" in content or "from torch" in content: return True
            if "import hypothesis" in content or "from hypothesis" in content: return True
            if "import PIL" in content or "from PIL" in content: return True
    except Exception as e:
        print(f"Warning: Failed to read {path} - {e}")
        raise
    return False

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    test_dir = os.path.join(base_dir, "test")
    
    ignore_args = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if should_ignore(full_path):
                ignore_args.append(f"--ignore={full_path}")

    ignore_dirs = ["models", "testextra", "null", "unit"]
    for d in ignore_dirs:
        ignore_args.append(f"--ignore={os.path.join(test_dir, d)}")
    
    # Use -n auto for parallel execution if xdist is available
    args = [test_dir] + ignore_args
    
    print(f"Running pytest with {len(ignore_args)} ignore args")
    sys.exit(pytest.main(args))
