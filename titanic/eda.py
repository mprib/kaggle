
#%%

import polars as pl
from pathlib import Path


#%% Read in data
data_directory = Path(Path(__file__).parent, "raw_data")

test_data = pl.read_csv(Path(data_directory,"test.csv"))
train_data = pl.read_csv(Path(data_directory,"train.csv"))
# %%
train_data.schema
# %%
train_data.describe()
# %%

# initial step is to parse the Name data...
def get_title(full_name:str)->list[str]:
    last_rest = full_name.split(",", maxsplit=1)
    # last_name = last_rest[0]
    title_rest = last_rest[1].split(".", maxsplit=1)
    title = title_rest[0].replace("the", "").replace(" ", "")
    return title

def set_title_group(title:str)->str:
    match title:
        case "Capt" | "Col" | "Major":
            return "Military"
        case "Lady" | "Jonkheer" | "Sir" | "Lord" | "Don":
            return "Aristocracy"
        case "Rev" | "Dr":
            return "Profession"
        case "Mrs" | "Mme":
            return "MarriedWoman"
        case "Miss" | "Mme":
            return "SingleWoman"

# %%
train_data = train_data.with_columns(pl.col("Name").apply(get_title).alias("Title"))
title_counts = train_data.group_by("Title").agg(pl.count("Name"))
title_counts
# %%
