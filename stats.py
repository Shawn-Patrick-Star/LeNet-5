import sys

STATS = {
    "epoch": 0,
    "train": {
        "loss": 0
    },
    "test": {
        "loss": 0,
        "accuracy": 0
    },
    "time": 0,
}
TEMPLATE_MSG = ("Epoch: {} \t Time: {:.2f} \t\t"
                "Train loss: {:.2f} \t\t"
                "Test loss:  {:.2f} / accuracy: {:.2f} \r")

def print_msg():
    sys.stdout.write('\033[2K\033[1G')
    msg = TEMPLATE_MSG.format(
        STATS["epoch"],
        STATS["time"],
        STATS["train"]["loss"],
        STATS["test"]["loss"], STATS["test"]["accuracy"]
    )
    sys.stdout.write(msg)
    
def update_stats(batch_time, train_loss, test_loss, accuracy):
    STATS["epoch"] += 1
    STATS["time"] += batch_time
    STATS["train"]["loss"] = train_loss
    STATS["test"]["loss"] = test_loss
    STATS["test"]["accuracy"] = accuracy
    print_msg()

def init_stats():
    STATS["epoch"] = 0
    STATS["time"] = 0
    STATS["train"]["loss"] = 0
    STATS["test"]["loss"] = 0
    STATS["test"]["accuracy"] = 0
    print_msg()
