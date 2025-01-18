import os
import sys
import base64
import json
import logging
import torch
from torch.utils.data import DataLoader
from .genericDataset import GenericDataset
from .quantUtils import evaluate_metrics, quantize_model_fx, quantize_model_dynamic
from .utils import load_class_from_file, parse_dynamic_args

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
        python_file_path = save_file_from_base64_string(files.get('pythonFile'), "model.py")
        model_weights_path = save_file_from_base64_string(files.get('modelFile'), "trained_model.pth")

        # Extract other parameters
        model_class = form.get("model_class")
        batch_size = int(form.get("batch_size", 32))
        num_batches = int(form.get("num_batches", 1))
        save_onnx_flag = form.get("save_onnx", "False").lower() == "true"
        evaluate_metrics_flag = form.get("evaluate_metrics", "False").lower() == "true"


        #static quantization can either be int8 or float16 so we need to check if we have for example int8,float16

        static_quantization = form.get("static_quantization")
        dynamic_quantization = form.get("dynamic_quantization")

        try:
            static_quantization_dtype = static_quantization.split(",")
        except:
            static_quantization_dtype = None
        
        try:
            dynamic_quantization_dtype = dynamic_quantization.split(",")
        except:
            dynamic_quantization_dtype = None



        dynamic_args = parse_dynamic_args(form.get("args"))
        logging.info("Parsed parameters.")


        logging.info(f"Shape of training set: {torch.load(training_set_path)['data'].shape}")
        logging.info(f"Shape of test set: {torch.load(test_set_path)['data'].shape}")
        
        #take shape from training set
        shape = torch.load(training_set_path)['data'][0].shape
        
        logging.info(f"Shape of data: {shape}")
        logging.info(f"Shape data type: {type(shape)}")
        # Load datasets
        train_ds = GenericDataset(training_set_path, shape)
        test_ds = GenericDataset(test_set_path, shape)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        logging.info("Datasets loaded.")

        # Load model definition
        model = load_class_from_file(model_class, os.path.dirname(python_file_path), python_file_path, **dynamic_args)
        model = model.to(DEVICE)
        logging.info("Model class loaded.")

        # Load model weights
        model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE, weights_only=False))
        logging.info("Model weights loaded.")

        if evaluate_metrics_flag:
            # Evaluate raw model
            raw_metrics = evaluate_metrics(test_loader, model)
            #model size in Mb
            model_size = os.path.getsize(model_weights_path) / (1024 * 1024)
            raw_metrics["model_size"] = model_size
            logging.info("Raw model evaluation completed.")
        else:
            raw_metrics = None


        quantized_models = {"static_quantization": {}, "dynamic_quantization": {}}

        logging.info(f"Model content: {raw_metrics}")

        if dynamic_quantization_dtype is not None: 
            for datatype in dynamic_quantization_dtype:
                if datatype == "int8":
                    quantize(model, train_loader, num_batches, torch.qint8, quantized_models, "dynamic_quantization")
                elif datatype == "float16":
                    quantize(model, train_loader, num_batches, torch.float16, quantized_models, "dynamic_quantization")
                else:
                    logging.error(f"Invalid dynamic quantization datatype: {datatype}. Skipping quantization.")

        if static_quantization_dtype is not None:
            for datatype in static_quantization_dtype:
                if datatype == "int8":
                    quantize(model, train_loader, num_batches, torch.qint8, quantized_models, "static_quantization")
                elif datatype == "float16":
                    quantize(model, train_loader, num_batches, torch.float16, quantized_models, "static_quantization")
                else:
                    logging.error(f"Invalid static quantization datatype: {datatype}. Skipping quantization.")




        # Save quantized model to disk
        for quantized_type, quantized_models_dict in quantized_models.items():
            for quantized_dtype, quantized_model in quantized_models_dict.items():
                quantized_model_path = f"quantized_model_{quantized_type}_{quantized_dtype}.pth"
                torch.save(quantized_model.state_dict(), quantized_model_path)


        # Encode quantized model as Base64
        quantized_model_base64 = {}

        for quantized_type, quantized_models_dict in quantized_models.items():
            quantized_model_base64[quantized_type] = {}
            for quantized_dtype, quantized_model in quantized_models_dict.items():
                quantized_model_path = f"quantized_model_{quantized_type}_{quantized_dtype}.pth"
                quantized_model_base64[quantized_type][quantized_dtype] = encode_file_to_base64(quantized_model_path)
                logging.info(f"Quantized model saved at: {quantized_model_path}")


        if evaluate_metrics_flag:
            # Evaluate quantized models
            quantized_metrics = {}
            for quantized_type, quantized_models_dict in quantized_models.items():
                quantized_metrics[quantized_type] = {}
                for quantized_dtype, quantized_model in quantized_models_dict.items():
                    metrics = evaluate_metrics(test_loader, quantized_model)
                    #measure model size in Mb
                    model_size = os.path.getsize(f"quantized_model_{quantized_type}_{quantized_dtype}.pth") / (1024 * 1024)
                    metrics["model_size"] = model_size
                    quantized_metrics[quantized_type][quantized_dtype] = metrics
                    logging.info(f"Quantized model evaluation completed for {quantized_type} ({quantized_dtype}).")

        else:
            quantized_metrics = None

        # Save ONNX file if required from quantized models

        onnx_base64 = save_onnx_for_all_models(quantized_models, test_loader, save_onnx_flag)

        # Return JSON response
        response = {
            "raw_metrics": raw_metrics,
            "quantized_metrics": quantized_metrics,
            "quantized_model_base64": quantized_model_base64,
            "onnx_base64": onnx_base64
        }
        logging.info("Execution completed. Response prepared.")
        return response
    except Exception as e:
        logging.error("Exception occurred in execute function.", exc_info=True)
        raise


def quantize(model, train_loader, num_batches, quantization_type, quantized_models, quantization_category):
    """Helper function to perform quantization and store results."""

    quantization_dt = "int8" if quantization_type == torch.qint8 else "float16"

    try:
        # Quantize the model
        if quantization_category == "dynamic_quantization":
            quantized_model = quantize_model_dynamic(model, train_loader, num_batches=num_batches, type=quantization_type)
            logging.info(f"Model quantized using dynamic quantization ({quantization_type}).")
        elif quantization_category == "static_quantization":
            quantized_model = quantize_model_fx(model, train_loader, num_batches=num_batches, type=quantization_type)
            logging.info(f"Model quantized using static quantization ({quantization_type}).")
        else:
            raise ValueError("Invalid quantization category")
        
        # Store the quantized model
        quantized_models[quantization_category][quantization_dt] = quantized_model
    except Exception as e:
        logging.error(f"Error during {quantization_category} ({quantization_type}) quantization: {e}")


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


def save_onnx(model, dummy_input, output_path):
    # Resetting the global export state before calling export
    torch.onnx._globals.in_onnx_export = False
    # Exporting the model
    torch.onnx.export(model, dummy_input, output_path)

def get_dummy_input_for_model(test_loader):
    # Fetch a batch from the test loader (e.g., first batch)
    batch_data, _ = next(iter(test_loader))

    # Check the shape of the data, assuming batch_data has shape [batch_size, channels, length]
    print(f"Original batch data shape: {batch_data.shape}")

    # Ensure the data is in the correct shape: [batch_size, channels, length] for 1D convolutions
    # If batch_data is 4D like [batch_size, channels, height, width], we might need to reshape it
    if len(batch_data.shape) == 4:  # e.g., for image data, batch_size, channels, height, width
        # Remove the extra dimension (height or width)
        batch_data = batch_data.squeeze(-1)  # Remove the last dimension (e.g., width or height)

    print(f"Reshaped batch data shape: {batch_data.shape}")

    return batch_data

def save_onnx_for_all_models(quantized_models, test_loader, save_onnx_flag):
    onnx_base64 = {}

    if save_onnx_flag:
        for quantized_type, quantized_models_dict in quantized_models.items():
            onnx_base64[quantized_type] = {}
            for quantized_dtype, quantized_model in quantized_models_dict.items():
                # Generate the ONNX file path
                onnx_path = f"model_{quantized_type}_{quantized_dtype}.onnx"
                
                # Get a properly shaped dummy input
                dummy_input = get_dummy_input_for_model(test_loader)

                try:
                    # Save the ONNX model
                    save_onnx(quantized_model, dummy_input, onnx_path)
                    # Encode the saved ONNX model into base64
                    onnx_base64[quantized_type][quantized_dtype] = encode_file_to_base64(onnx_path)
                    logging.info(f"ONNX model saved at: {onnx_path}")
                except Exception as e:
                    logging.error(f"Error saving ONNX model for {quantized_type} {quantized_dtype}: {e}")

    return onnx_base64
