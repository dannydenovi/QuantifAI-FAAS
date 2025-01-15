import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import base64
import json
import logging
import torch
from torch.utils.data import DataLoader
from genericDataset import GenericDataset
from quantUtils import evaluate_metrics, quantize_model_fx, quantize_model_dynamic
from utils import load_class_from_file, parse_dynamic_args, save_onnx

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")



# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device set to: {DEVICE}")


def handle(event, context):
    """
    OpenFaaS entry point for handling function requests with two parameters.
    
    Args:
        event (dict): Input event containing request details (e.g., body, headers).
        context (dict): Contextual information (e.g., function name, environment details).
    
    Returns:
        dict: HTTP response containing a status code and response body.
    """
    logging.info("Handle function started.")
    try:
        # Extract body from event
        if isinstance(event, dict):
            event_body = event.get("body", "")
        elif hasattr(event, "body"):
            event_body = event.body
        else:
            raise ValueError("Unsupported event format. Expected dict or object with 'body' attribute.")

        #logging.debug(f"Event body: {event_body}")

        # Use the context for additional information
        function_name = context.get("function_name", "unknown") if isinstance(context, dict) else getattr(context, "function_name", "unknown")
        logging.info(f"Function context: {function_name}")

        # Parse the event body as JSON
        request_data = json.loads(event_body)
        #logging.debug(f"Parsed request data: {request_data}")

        # Process the request
        response = execute(request_data)

        # Add contextual information to the response
        response["function_name"] = function_name

        logging.info("Execution completed successfully.")
        return {
            "statusCode": 200,
            "body": json.dumps(response)
        }
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON input.", exc_info=True)
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON format", "details": str(e)})
        }
    except Exception as e:
        logging.error("Unhandled exception in handle function.", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

def execute(req: dict) -> dict:
    """
    Main logic to process the incoming request and return results as JSON.
    
    Args:
        req (dict): JSON request containing form data and Base64-encoded files.
    
    Returns:
        dict: Response containing metrics and model details.
    """
    try:
        logging.info("Starting execution.")

        # Parse form data
        form = req.get('form', {})
        files = req.get('files', {})
        logging.debug(f"Form data: {form}")
        logging.debug(f"Files: {list(files.keys())}")

        # Save datasets and model files
        training_set_path = save_file_from_base64_string(files.get('trainingSetFile'), "training_dataset.pth")
        test_set_path = save_file_from_base64_string(files.get('testSetFile'), "test_dataset.pth")
        python_file_path = save_file(files.get('pythonFile'), "model.py")
        model_weights_path = save_file(files.get('modelFile'), "trained_model.pth")

        # Extract other parameters
        model_class = form.get("model_class")
        batch_size = int(form.get("batch_size", 32))
        input_format = eval(form.get("input_format")) if form.get("input_format") else None
        num_batches = int(form.get("num_batches", 1))
        is_classification = form.get("is_classification", "False").lower() == "true"
        save_onnx_flag = form.get("save_onnx", "False").lower() == "true"
        evaluate_metrics_flag = form.get("evaluate_metrics", "False").lower() == "true"
        quantization_mode = form.get("quantization_mode", "static")
        quantization_type = form.get("quantization_type", "int8")

        quantization_mode = "static" if quantization_mode not in ["static", "dynamic"] else quantization_mode

        if quantization_type == "int8":
            quantization_type = torch.qint8
        else:
            quantization_type = torch.float16
        

        dynamic_args = parse_dynamic_args(form.get("args"))
        logging.info("Parsed parameters.")

        # Load datasets
        train_ds = GenericDataset(training_set_path, input_format)
        test_ds = GenericDataset(test_set_path, input_format)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        logging.info("Datasets loaded.")

        # Load model definition
        model = load_class_from_file(model_class, os.path.dirname(python_file_path), python_file_path, **dynamic_args)
        model = model.to(DEVICE)
        logging.info("Model class loaded.")

        # Load model weights
        model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
        logging.info("Model weights loaded.")

        if evaluate_metrics_flag:
            # Evaluate raw model
            raw_metrics = evaluate_metrics(test_loader, model, is_classification=is_classification)
            logging.info("Raw model evaluation completed.")
        else:
            raw_metrics = None

        # Quantize model
        if quantization_mode == "dynamic":
            logging.info("Dynamic quantization...")
            quantized_model = quantize_model_dynamic(model, train_loader, num_batches, type=quantization_type)
        else:
            logging.info("Static quantization...")
            quantized_model = quantize_model_fx(model, train_loader, num_batches, type=quantization_type)

        quantized_model_path = "quantized_model.pth"
        torch.save(quantized_model.state_dict(), quantized_model_path)
        logging.info(f"Quantized model saved at: {quantized_model_path}")

        # Encode quantized model as Base64
        quantized_model_base64 = encode_file_to_base64(quantized_model_path)

        if evaluate_metrics_flag:
            # Evaluate quantized model
            quantized_metrics = evaluate_metrics(test_loader, quantized_model, is_classification=is_classification)
            logging.info("Quantized model evaluation completed.")
        else:
            quantized_metrics = None

        # Save ONNX file if required
        onnx_base64 = None
        if save_onnx_flag:
            onnx_path = "quantized_model.onnx"
            save_onnx(quantized_model, test_loader, onnx_path)
            onnx_base64 = encode_file_to_base64(onnx_path)
            logging.info(f"ONNX file saved at: {onnx_path}")

        # Return JSON response
        response = {
            "raw_metrics": raw_metrics,
            "quantized_metrics": quantized_metrics,
            "raw_model_size_mb": os.path.getsize(model_weights_path) / 1024**2,
            "quantized_model_size_mb": os.path.getsize(quantized_model_path) / 1024**2,
            "quantized_model_base64": quantized_model_base64,
            "onnx_base64": onnx_base64,
        }
        logging.info("Execution completed. Response prepared.")
        return response
    except Exception as e:
        logging.error("Exception occurred in execute function.", exc_info=True)
        raise


def save_file_from_base64_string(file_data, default_filename):
    """
    Save a Base64-encoded file string to the local disk.
    """
    if not file_data:
        raise ValueError(f"File data missing for {default_filename}")

    filename = file_data.get('filename', default_filename)
    base64_content = file_data.get('content', '')

    if not base64_content:
        raise ValueError(f"Base64 content is missing for {filename}")

    file_path = os.path.join("/tmp", filename)
    os.makedirs("/tmp", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(base64.b64decode(base64_content))

    return file_path


def save_file(file_data, default_filename):
    """
    Save a Base64-encoded file to the local disk.
    """
    if not file_data:
        return default_filename

    filename = file_data.get('filename', default_filename)
    content = base64.b64decode(file_data.get('content', ''))
    file_path = os.path.join("/tmp", filename)

    os.makedirs("/tmp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)

    return file_path


def encode_file_to_base64(file_path):
    """
    Encode a file to Base64 format.
    """
    if not os.path.exists(file_path):
        return None

    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def jsonify(response):
    """
    Converts a dictionary into a JSON response string.
    """
    return json.dumps(response)
