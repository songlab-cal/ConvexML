import pickle


def read_ble(
    ble_path: str,
):
    with open(ble_path, "rb") as f:
        return pickle.load(f)


def write_ble(
    ble,
    ble_path: str,
):
    with open(ble_path, "wb") as f:
        pickle.dump(ble, f)
        f.flush()
