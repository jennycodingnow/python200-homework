import pandas as pd

files_list = [
    "data/world_happiness_2015.csv",
    "data/world_happiness_2016.csv",
    "data/world_happiness_2017.csv",
    "data/world_happiness_2018.csv",
    "data/world_happiness_2019.csv",
    "data/world_happiness_2020.csv",
    "data/world_happiness_2021.csv",
    "data/world_happiness_2022.csv",
    "data/world_happiness_2023.csv",
    "data/world_happiness_2024.csv"
]


# Load temporarily for inspection
raw_files = []

for file in files_list:
    df = pd.read_csv(file, sep=";", decimal=",")
    raw_files.append(df)


# Check column names
for i, df in enumerate(raw_files, start=2015):
    print(f"{i} columns:")
    print(df.columns.tolist())
    print()


same_columns = all(
    df.columns.equals(raw_files[0].columns)
    for df in raw_files
)

print("Same columns:", same_columns)

for i, df in enumerate(raw_files, start=2015):
    if not df.columns.equals(raw_files[0].columns):
        print(f"{i} has different columns")