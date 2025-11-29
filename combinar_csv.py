import gzip

BUCKET= "/home/guijomter/buckets/b1/Compe_03/datasets"
csv_file_1 = f"{BUCKET}/competencia_02_crudo.csv.gz"
csv_file_2 = f"{BUCKET}/competencia_03_crudo.csv.gz"
csv_file_out = f"{BUCKET}/competencia_03_crudo_comp.csv.gz"

with gzip.open(csv_file_out, "wt") as fout:
    with gzip.open(csv_file_1, "rt") as f1:
        fout.write(f1.read())

    with gzip.open(csv_file_2, "rt") as f2:
        # Saltar el header del segundo archivo
        next(f2)
        fout.write(f2.read())