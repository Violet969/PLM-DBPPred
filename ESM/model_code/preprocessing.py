import sys

def read_fasta(path):
    # load file
    try:
        infile = open(path, "r")
    except IOError as err:
        print("Can't open file: " + err)
        sys.exit(1)

    header = None
    body = str()

    # parse fasta file
    for line in infile:
        get_header = line.find(">")

        if get_header == -1:
            body += line.strip()
        else:
            if len(body) != 0:
                yield header, body.upper()
                body = str()
            header = line.strip()

    yield header, body.upper()

    infile.close()