import json

# get closest string
import difflib

# converts between champion names, ids and indices

class ChampionConverter:
    def __init__(self):
        self.champion_names = []
        self.champion_ids = []
        self.champion_indices = {}
        self.champion_names_to_ids = {}
        self.champion_ids_to_names = {}
        self.champion_names_to_indices = {}
        self.champion_indices_to_names = {}
        self.champion_ids_to_indices = {}
        self.champion_indices_to_ids = {}
        self.load_champion_data()

    def load_champion_data(self):
        with open("champion.json", "r", encoding="utf-8") as f:
            champion_data = json.load(f)
        for champion in champion_data["data"].values():
            self.champion_names.append(champion["id"])
            self.champion_ids.append(int(champion["key"]))
        names_and_ids = zip(self.champion_names, self.champion_ids)
        names_and_ids = sorted(names_and_ids, key=lambda x: x[1])
        self.champion_names, self.champion_ids = zip(*names_and_ids)
        for i, champion_name in enumerate(self.champion_names):
            self.champion_indices[champion_name] = i
            self.champion_names_to_ids[champion_name] = self.champion_ids[i]
            self.champion_names_to_indices[champion_name] = i
            self.champion_indices_to_names[i] = champion_name
        for i, champion_id in enumerate(self.champion_ids):
            self.champion_ids_to_names[champion_id] = self.champion_names[i]
            self.champion_ids_to_indices[champion_id] = i
            self.champion_indices_to_ids[i] = champion_id


    def get_champion_name_from_index(self, champion_index):
        return self.champion_indices_to_names[champion_index]

    def get_champion_id_from_index(self, champion_index):
        return self.champion_indices_to_ids[champion_index]

    def get_champion_index_from_id(self, champion_id):
        return self.champion_ids_to_indices[champion_id]

    def get_champion_index_from_name(self, champion_name):
        return self.champion_names_to_indices[champion_name]
    
    def get_champion_id_from_name(self, champion_name):
        return self.champion_names_to_ids[champion_name]

    def get_champion_name_from_id(self, champion_id):
        return self.champion_ids_to_names[champion_id]
    
    def get_closest_champion_name(self, champion_name):
        matches = difflib.get_close_matches(champion_name, self.champion_names, 1)
        if len(matches) == 0:
            return None
        return matches[0]
    

if __name__ == "__main__":
    conv = ChampionConverter()
    while True:
        champion_name = input("Enter champion name: ")
        champ = conv.get_closest_champion_name(champion_name)
        if champ is None:
            print("No champion found!")
            continue
        print(champion_name, "->", champ)
        # id
        print("ID:", conv.get_champion_id_from_name(champ))
        # index
        print("Index:", conv.get_champion_index_from_name(champ))
        print()