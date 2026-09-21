import json

problems_to_remove = {
    "2003_IMO_Problems_6", "1984_IMO_Problems_5", "2009_IMO_Problems_2", "1971_IMO_Problems_5",
    "1964_IMO_Problems_2", "1970_IMO_Problems_6", "1985_IMO_Problems_2", "2006_IMO_Problems_5",
    "1970_IMO_Problems_5", "2001_IMO_Problems_3", "1960_IMO_Problems_4", "1985_IMO_Problems_4",
    "2007_IMO_Problems_1", "1971_IMO_Problems_3", "2014_IMO_Problems_1", "1965_IMO_Problems_2",
    "1966_IMO_Problems_3", "1974_IMO_Problems_2", "2001_IMO_Problems_1", "1959_IMO_Problems_6",
    "1976_IMO_Problems_5", "1974_IMO_Problems_3", "1971_IMO_Problems_4", "1977_IMO_Problems_4",
    "1982_IMO_Problems_4", "2019_IMO_Problems_2", "1987_IMO_Problems_5"
}

# Load the dataset
file_path = "data/processed/creative_math.json"
with open(file_path, "r") as f:
    original_data = json.load(f)

# Filter the dataset
filtered_data = [
    entry for entry in original_data
    if entry["meta_data"]["id"] not in problems_to_remove
]

# Save the filtered dataset
filtered_path = "creative_math.json"
with open(filtered_path, "w") as f:
    json.dump(filtered_data, f, indent=4)

filtered_path
