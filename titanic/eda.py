
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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
        case "Lady" | "Jonkheer" | "Sir" | "Lord" | "Don" | "Countess" | "Count":
            return "Aristocracy"
        case "Rev" | "Dr":
            return "Profession"
        case "Mrs" | "Mme" | "Mlle":
            return "MarriedWoman"
        case "Miss" | "Mme" | "Ms":
            return "SingleWoman"
        case "Mr":
            return "OlderMan"
        case "Master":
            return "YoungerMan"

def get_cabin_level(cabin:str)->str:
    if cabin == "null":
        return "None"
    else:
        return cabin[0]



train_data = (train_data
              .with_columns(pl.col("Name").apply(get_title).alias("Title"))
              .with_columns(pl.col("Title").apply(set_title_group).alias("Title_Group"))
              .with_columns(pl.col("Cabin").fill_null("null"))
              .with_columns(pl.col("Cabin").apply(get_cabin_level).alias("Cabin_Level"))
)

title_counts = train_data.group_by(["Title_Group"]).agg(pl.count("Name"))
title_counts

cabin_levels = train_data.group_by(["Cabin_Level"]).agg(pl.count("Name"))
cabin_levels
# %%

X = train_data.select([
                    'Pclass',
                    'Sex',
                    'Age',
                    'SibSp',
                    'Parch',
                    'Fare',
                    'Embarked',
                    'Title_Group'
                ]).to_pandas()

y = train_data.select("Survived").to_pandas()

# %%

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model and fit it to your training data
model = LinearRegression()
model.fit(X_train, y_train)

# Use your trained model to make predictions on your test data
y_pred = model.predict(X_test)

# Evaluate your model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')