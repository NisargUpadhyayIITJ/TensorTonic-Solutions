def early_stopping(val_losses: list, patience: int) -> dict:
    """
    Returns the best and stopping epoch indices in a dictionary.
    """
    counter = 0
    prev_loss = 1e9
    best_epoch = 0
    stop_epoch = len(val_losses) - 1
    for epoch, curr_loss in enumerate(val_losses):
        if(prev_loss > curr_loss):
            prev_loss = curr_loss
            best_epoch = epoch
            counter = 0

        else:
            counter += 1
            if(counter >= patience):
                stop_epoch = epoch
                return {
                    "best_epoch": best_epoch,
                    "stop_epoch": stop_epoch
                }

    return {
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch
    }
        