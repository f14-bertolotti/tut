import inspect
import json
import os

def safe_convert(obj):
    """Converts an object to a JSON-serializable form, or a string if it fails."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)

def save_state(path: str):
    """Saves the current state of global and local variables to a JSON file."""
    
    frame = inspect.currentframe().f_back  # Get the caller's frame
    local_vars = frame.f_locals
    global_vars = frame.f_globals
    global_vars = {k: v for k, v in global_vars.items() if not k.startswith("__") and not inspect.ismodule(v)}

    state = {
        "globals": {k: safe_convert(v) for k, v in global_vars.items()},
        "locals" : {k: safe_convert(v) for k, v in  local_vars.items()}
    }

    with open(os.path.join(path), "w") as f:
        json.dump(state, f, indent=4)

