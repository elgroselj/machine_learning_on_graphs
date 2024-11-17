import pickle

def save_dataset(dataset, name="rel-amazon"):
    # dataset = get_dataset(name=name, download=True)
    with open(f"big_files/{name}.pkl", "wb") as f:
        pickle.dump(dataset, f)
        
def load_dataset(name="rel-amazon"):
    with open(f"big_files/{name}.pkl", "rb") as f:
        dataset = pickle.load(f)
    return dataset

# save_dataset(dataset)
dataset = load_dataset()


db = dataset.get_db()


def save_db(dataset, name="rel-amazon-db"):
    # dataset = get_dataset(name=name, download=True)
    with open(f"big_files/{name}.pkl", "wb") as f:
        pickle.dump(dataset, f)
        
def load_db(name="rel-amazon-db"):
    with open(f"big_files/{name}.pkl", "rb") as f:
        dataset = pickle.load(f)
    return dataset

save_db(db)
# db = load_db()
