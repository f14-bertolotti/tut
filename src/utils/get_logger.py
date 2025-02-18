import logging
import json

def get_logger(path):
    """
    Creates a logger that logs JSONL-formatted messages to the specified file.
    
    :param path: Path to the log file.
    :return: Configured logger.
    """
    # Create a logger
    logger = logging.getLogger(path)
    logger.setLevel(logging.INFO)

    # Create a file handler that writes to the specified path
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)

    # Define a custom formatter to log JSONL
    class JsonLineFormatter(logging.Formatter):
        def format(self, record):
            if isinstance(record.msg, dict):
                return json.dumps(record.msg)  # Convert dictionary to JSON string
            return super().format(record)

    # Set the custom formatter for the file handler
    file_handler.setFormatter(JsonLineFormatter())

    # Add the handler to the logger
    logger.addHandler(file_handler)

    # Ensure duplicate handlers are not added
    logger.propagate = False

    return logger
