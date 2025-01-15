import torch
from torch import nn
from torch.quantization import quantize_fx, QConfig, MinMaxObserver, PerChannelMinMaxObserver
from torch.ao.quantization.qconfig_mapping import QConfigMapping


# Metrics Calculation
def evaluate_metrics(test_dataloader, model, is_classification=False):
    """Evaluate all metrics in one pass."""
    model.eval()
    total, correct, total_loss, predictions, true_labels = 0, 0, 0.0, [], []
    loss_fn = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

    with torch.no_grad():
        for data, labels in test_dataloader:
            outputs = model(data)
            if is_classification:
                total_loss += loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
            else:
                total_loss += loss_fn(outputs.squeeze(), labels).item()
                predictions.append(outputs.squeeze())
                true_labels.append(labels)

            total += labels.size(0)

    results = {"loss": total_loss / total}
    if is_classification:
        results["accuracy"] = correct / total
    else:
        predictions = torch.cat(predictions)
        true_labels = torch.cat(true_labels)
        ss_residual = torch.sum((true_labels - predictions) ** 2).item()
        ss_total = torch.sum((true_labels - torch.mean(true_labels)) ** 2).item()
        results.update({
            "mse": total_loss / total,
            "r2": 1 - (ss_residual / ss_total),
            "mae": torch.mean(torch.abs(true_labels - predictions)).item()
        })
    return results


# Quantization
def quantize_model_fx(model, training_dataloader, num_batches=1, type=torch.qint8):
    """Quantize a model using FX Graph Mode."""
    model.eval()
    qconfig = QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8 if type == torch.qint8 else torch.float16),
        weight=PerChannelMinMaxObserver.with_args(dtype=type)
    )
    model.qconfig = qconfig
    example_inputs = next(iter(training_dataloader))[0]
    prepared_model = quantize_fx.prepare_fx(model, QConfigMapping().set_global(torch.quantization.default_qconfig), example_inputs)
    for i, (batch_data, _) in enumerate(training_dataloader):
        prepared_model(batch_data)
        if i >= num_batches - 1:
            break
    return quantize_fx.convert_fx(prepared_model)


def quantize_model_dynamic(model, training_dataloader, num_batches=1, type=torch.qint8):
    """Quantize a model using Dynamic Quantization."""
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # Modello da quantizzare
        {nn.Linear, nn.RNNCell, nn.GRUCell, nn.LSTMCell},  # Tipi di livelli da quantizzare dinamicamente
        dtype=type # Tipo di quantizzazione (opzionale, default è qint8)
    )


    return quantized_model

