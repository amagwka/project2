from torch.utils.tensorboard import SummaryWriter

_writer = SummaryWriter()

def log_scalar(tag: str, value: float, step: int):
    """Log a scalar value to TensorBoard."""
    _writer.add_scalar(tag, value, step)
