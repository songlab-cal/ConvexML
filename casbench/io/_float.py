def read_float(float_path: float):
    with open(float_path, "r") as input_file:
        return float(input_file.read())


def write_float(f: str, float_path: str):
    with open(float_path, "w") as output_file:
        output_file.write(str(f))
