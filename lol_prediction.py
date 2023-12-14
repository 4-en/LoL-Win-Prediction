import numpy as np
import champion_dicts
import tensorflow as tf
# this class is used with a model to perform different predictions based on game data

class LoLPredictor:
    def __init__(self, model):
        self.model = model
        self.converter = champion_dicts.ChampionConverter()

    def convert_to_input(self, blue_team, red_team, format="names"):
        """
        Converts the inputs if this functions to inputs for the model.

        Format = names | ids | idx
        """

        if len(blue_team) > 5 or len(red_team) > 5:
            raise Exception("teams can't have more than 5 members")

        if format == "names":
            blue_team = self.converter.get_champion_index_from_name(blue_team)
            red_team = self.converter.get_champion_index_from_name(red_team)

        elif format == "ids":
            blue_team = self.converter.get_champion_index_from_id(blue_team)
            red_team = self.converter.get_champion_index_from_id(red_team)
        
        elif format == "idx":
            # all good
            pass
        else:
            raise Exception("Invalid Argument. Format has to be 'names', 'ids', or 'idx'")

        # pad with zeroes
        while len(blue_team)<5:
            blue_team.append(0)
        while len(red_team)<5:
            red_team.append(0)

        champs = blue_team + red_team
        champs = np.array(champs)

        return champs


    def win_chance(self, blue_team, red_team, format="names") -> float:
        """
        Format = names | ids | idx
        """

        champs = self.convert_to_input(blue_team=blue_team, red_team=red_team, format=format)

        champs = np.reshape(champs, (1, -1))

        #print(champs)

        wr = self.model.predict(champs)
        wr = np.reshape(wr, (-1,))
        #print(wr)
        wr = float(wr)
        return wr
    
    def best_pick(self, blue_team, red_team, available = [], format="names", print_details=True) -> str:
        """
        Find the champion with the best win chance for the next pick.
        Tries only champions in available, unless len(available)==0.


        Format = names | ids | idx
        """

        if len(red_team)>=5 and red_team[4]!=0:
            raise Exception("Red team can't have 5 members to predict the next champion.")

        if len(available)>0:
            if format == "names":
                available = self.converter.get_champion_index_from_name(available)

            elif format == "ids":
                available = self.converter.get_champion_index_from_id(available)
        else:
            available = self.converter.get_all_indices()

        

        champs = self.convert_to_input(blue_team=blue_team, red_team=red_team, format=format)

        # remove champions that were already picked
        available = list(filter(lambda i: i not in champs, available))
        available = np.array(available)


        # find which position should be replaced
        order_to_pos = [0, 5, 6, 1, 2, 7, 8, 3, 4, 9]
        pos = -1
        for i in range(10):
            idx = order_to_pos[i]
            if champs[idx] == 0:
                pos = idx
                break

        if pos == -1:
            raise Exception("Failed to find empty position...")
        
        user_blue = pos < 5

        

        matches = np.tile(champs, len(available))
        matches = matches.reshape((-1, 10))
        matches[:, pos] = available

        results = self.model.predict(matches)

        # sorts all_champs based on results
        results = results.reshape((-1,))
        if not user_blue:
            # since the model only predict wr for blue side, wr is 1-res if user is on red side
            results = 1 - results
        indicies = np.argsort(results)


        indicies = indicies[::-1]

        available = available[indicies]
        results = results[indicies]


        if print_details:
            blue = [ self.converter.get_champion_name_from_index(i) for i in champs[:5]]
            red = [ self.converter.get_champion_name_from_index(i) for i in champs[5:]]

            print("5 best picks for")
            print(blue)
            print("vs")
            print(red)
            for i in range(min(len(available), 5)):
                wr = results[i]
                wr_rounded = round(float(wr) * 100, 2)
                champ = available[i]
                champ_name = self.converter.get_champion_name_from_index(int(champ))
                print(f"{champ_name}: {wr_rounded}%")



        ret = [ (self.converter.get_champion_name_from_index(i[0]), i[1]) for i in zip(available, results) ]
        return ret

def load_default_model():
    from models.lol_transformer import LoLTransformer

    print("Loading LoLTransformer_8b_12h...")
    model = LoLTransformer(8,12)

    test_in = tf.random.uniform(shape=(8,10), minval=0, maxval=169, dtype=tf.int32)
    _ = model.predict(test_in)


    model.load_weights("models/lol_transformer_8_12.h5")
    print("Loaded!")
    return model

        
if __name__ == "__main__":
    model = load_default_model()

    pred = LoLPredictor(model)

    chance = pred.win_chance(["Ahri", "Elise", "Singed"], ["DrMundo", "Kassadin"])
    print(chance)

    _ = pred.best_pick(["Ahri", "Elise", "Singed"], ["DrMundo", "Kassadin"])

    _ = pred.best_pick(["Ahri", "Elise", "Singed"], ["DrMundo", "Draven"], available=["Taliyah", "Yone", "Orianna"])

    
        