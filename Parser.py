import io

DELIMITER = "#\n"

def parse_instance(file_path):
    global DELIMITER

    lines, columns = [], []
    with io.open(file_path, encoding="utf8") as file:
        for line in file:
            if line != DELIMITER:
                lines.append(parse_line(line))
            else:
                break
        for line in file:
            columns.append(parse_line(line))
    return lines, columns

def parse_line(line):
    """ Parses a line containing integers separated by a space, or an
     empty line, then returns a list of integers, or an empty list.
    """
    return list(map(int, line.split()))