import gzip
import polars as pl
import shutil


BUCKET= "/home/guijomter/buckets/b1/datasets"
csv_file_1 = f"{BUCKET}/competencia_02_crudo.csv.gz"
csv_file_2 = f"{BUCKET}/competencia_03_crudo.csv.gz"
csv_file_out = f"{BUCKET}/competencia_03_crudo_comp.csv.gz"
csv_file_out_uncompressed = f"{BUCKET}/competencia_03_crudo_comp.csv"

df1 = pl.read_csv(csv_file_1, infer_schema_length=None , try_parse_dates=False)
df2 = pl.read_csv(csv_file_2, infer_schema_length=None , try_parse_dates=False)

# Concatenar
df = pl.concat([df1, df2], how="vertical")

# Guardar limpio
df.write_csv(csv_file_out_uncompressed)


with open(csv_file_out_uncompressed, "rb") as f_in:
    with gzip.open(csv_file_out, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)





# with gzip.open(csv_file_out, "wt") as fout:
#     with gzip.open(csv_file_1, "rt") as f1:
#         fout.write(f1.read())

#     with gzip.open(csv_file_2, "rt") as f2:
#         # Saltar el header del segundo archivo
#         next(f2)
#         fout.write(f2.read())