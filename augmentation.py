# different augmentation methods for lol match data

from champion_dicts import ChampionConverter
import numpy as np

from keras.utils import Sequence


class MatchAugmentation(Sequence):

    def __init__(self, data, labels, batch_size=32, aug_chance=0.6, adjust_labels=False, mask_champs=True, max_replace=1, swap_teams=False):
        self.champion_converter = ChampionConverter()
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.aug_chance = aug_chance
        self.mask_champs = mask_champs
        if max_replace > 10:
            max_replace = 10
        self.max_replace = max_replace
        self.swap_teams = swap_teams

        # TODO: implement label adjustment
        self.adjust_labels = adjust_labels # if true, labels are moved towards 0.5 depending on amount of augmentations

    def __getitem__(self, idx):
        matches = []
        labels = []
        batch_i = 0
        while batch_i < self.batch_size:
            match, res = self.data[idx * self.batch_size + batch_i], self.labels[idx * self.batch_size + batch_i]

            match, res = self.augment_match(match, res)
            matches.append(match)
            labels.append(res)
            
            batch_i += 1

        matches = np.array(matches)
        labels = np.array(labels)
        return matches, labels

    
    def __len__(self):
        return len(self.data) // self.batch_size
    
    def augment_match(self, match, res):
        match = match.copy()
        res = res.copy()
        #print("Before", match, res)
        match, res = self.shuffle_match(match, res)
        #print("After", match, res)

        if np.random.rand() < self.aug_chance:
            if np.random.rand() < 0.5 and self.max_replace > 0:
                match, res = self.replace_champion(match, res)
            elif self.mask_champs:
                match, res = self.partial_select(match, res)
            else:
                # do nothing
                pass
        #match = [int(x) for x in match]
        #match = [self.champion_converter.get_champion_index_from_id(x) for x in match]
        return match, res
    
    def shuffle_match(self, match, label):
        "shuffles the champions in each team"
        team1 = match[:5]
        team2 = match[5:]
        np.random.shuffle(team1)
        np.random.shuffle(team2)

        # 0.5 chance to swap teams
        # this makes side difference irrelevant and only focuses on champion difference
        # arguable if this is a good thing
        if self.swap_teams and np.random.rand() < 0.5:
            # invert labels
            return np.concatenate((team2, team1)), 1-label

        return np.concatenate((team1, team2)), label
    
    def replace_champion(self, match, label):
        """replaces a random champion with a random champion
        Since changing one champion is unlikely to change the winchance by much, and even if it does
        the favoured team could still lose, we can replace a random champion with a random champion to generate more samples.
        It is more likely that the result is still correct than not."""

        # random amount of replacements [1, self.max_replace]
        replacements = 1
        if self.max_replace > 1:
            replacements = np.random.randint(1, self.max_replace+1)

        # determine positions that are replaces, don't replace same pos twice
        positions = [ i for i in range(10)]
        np.random.shuffle(positions)

        for i in range(replacements):
            replace_idx = positions[i]
            new_id = np.random.randint(1, self.champion_converter.champion_count)

            # dont insert id that is already in the match, since this is not possible in ranked matches
            while new_id in match:
                new_id = np.random.randint(1, self.champion_converter.champion_count)
            match[replace_idx] = new_id
        return match, label

    def partial_select(self, match, label):
        """set an amount of champions to null
        This is to simulate a champion select where not all champions have been picked yet.
        Since this is a valid champion select state of the full match, this state should also have the same win value, even if
        the win chance is not exactly the same.
        Since multiple different matches can lead to the same partial select state, we can generate more samples this way."""
        # pick order in draft (b1, r1, r2, b2,...) -> pick index in match
        pick_order = [0,3,4,7,8,1,2,5,6,9]
        remove_num = np.random.randint(0, len(match))
        mask = np.where(np.array(pick_order) < remove_num, 1, 0)
        match = match * mask
        return match, label
    