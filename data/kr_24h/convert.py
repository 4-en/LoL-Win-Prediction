

# no,gameNo,playerNo,CreationTime,KoreanTime,participantId,teamId,summonerName,gameEndedInEarlySurrender,gameEndedInSurrender,teamEarlySurrendered,win,teamPosition,kills,deaths,assists,objectivesStolen,visionScore,puuid,summonerId,baronKills,bountyLevel,champLevel,championName,damageDealtToBuildings,damageDealtToObjectives,detectorWardsPlaced,doubleKills,dragonKills,firstBloodAssist,firstBloodKill,firstTowerAssist,firstTowerKill,goldEarned,inhibitorKills,inhibitorTakedowns,inhibitorsLost,killingSprees,largestKillingSpree,largestMultiKill,longestTimeSpentLiving,neutralMinionsKilled,objectivesStolenAssists,pentaKills,quadraKills,timeCCingOthers,timePlayed,totalDamageDealt,totalDamageDealtToChampions,totalDamageTaken,totalHeal,totalHealsOnTeammates,totalMinionsKilled,totalTimeCCDealt,totalTimeSpentDead,totalUnitsHealed,tripleKills,unrealKills

# encoding: cp949
import csv
# loading bar
import tqdm
import os
import numpy as np
import zipfile
import urllib.request

from champion_dicts import ChampionConverter


TARGET_NAME = "kr_soloq_24h.zip"
URL = "https://www.kaggle.com/datasets/junhachoi/all-ranked-solo-games-on-kr-server-24-hours/download?datasetVersionNumber=1"
TARGET_DIR = TARGET_NAME.split(".")[0]
FILENAME = "sat df.csv"
file_path = os.path.join(TARGET_DIR, FILENAME)

converter = ChampionConverter()


CLEAN_MATCHES = True
MIN_LEVEL = 7
MIN_AVG_LEVEL = 10
MIN_LEVEL_DIFF = 4

def filter_player(player) -> bool:
    """
    Returns true if a player data is above threshold and match should be removed from list
    """

    # gameEndedInEarlySurrender
    if player["gameEndedInEarlySurrender"] == "True":
        return True
    
    timePlayed = player["timePlayed"]
    timePlayed = int(timePlayed) / 60.0
    if timePlayed < 20:
        return True

    return False

def split_iterable(data, weights=(1,1)) -> list:
    # splits an iterable into multiple iterables based on weights
    # weights should be a tuple of ints
    # returns a list of iterables
    ret = []
    total = sum(weights)
    before = 0
    for weight in weights:
        ret.append(data[int(before/total*len(data)):int((before+weight)/total*len(data))])
        before += weight
    return ret


def convert_game(game, filter_matches=CLEAN_MATCHES)->list[int]:
    # converts a game to a list of ints
    # first 5 are blue champ ids, next 5 are red champ ids, last one is 1 if blue won, 0 if red won
    blue_team = []
    red_team = []
    blue_win = None
    levels = []
    for player in game:
        if filter_matches and filter_player(player):
            return None
            raise Exception("Player is below threshold!")
        champion_name = player["championName"]
        level = int(player["champLevel"])
        levels.append(level)
        champion_id = -1
        while champion_id == -1:
            try:
                champion_id = converter.get_champion_id_from_name(champion_name)
            except:
                champion_name = converter.get_closest_champion_name(champion_name)
                if champion_name is None:
                    return None
                    raise Exception("Champion name not found!")
        if player["teamId"] == "100":
            blue_team.append(champion_id)
            if blue_win is None:
                blue_win = player["win"] == "True"
            else:
                if blue_win != (player["win"] == "True"):
                    return None
                    raise Exception("Blue win is not consistent!")
        else:
            red_team.append(champion_id)
            if blue_win is None:
                blue_win = player["win"] == "False"
            else:
                if blue_win == (player["win"] == "True"):
                    return None
                    raise Exception("Blue win is not consistent!")
                
    if filter_matches:
        avg_level = sum(levels)/len(levels)
        if avg_level < MIN_AVG_LEVEL:
            return None
            raise Exception("Average level is below 10!")
        lowest_level = min(levels)

        # try to remove games with afk players
        if lowest_level < MIN_LEVEL:
            return None
            raise Exception("Lowest level is below 7!")
        if lowest_level < avg_level-MIN_LEVEL_DIFF:
            return None
            raise Exception("Lowest level is more than 3 below average level!")
    
    both_teams = blue_team+red_team
    both_teams.append(1 if blue_win else 0)
    return both_teams

def save_csv(data):
    filename = "game_data.csv"
    header = ["blue1", "blue2", "blue3", "blue4", "blue5", "red1", "red2", "red3", "red4", "red5", "blue_win"]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)

def load_raw_csv(file_path = file_path) -> dict:
    games = {}
    print("Loading game data...")
    with open(file_path, "r", encoding="cp949") as f:
        reader = csv.DictReader(f)
        for row in tqdm.tqdm(reader):
            gameNo = row["gameNo"]
            if gameNo not in games:
                games[gameNo] = []
            games[gameNo].append(row)

    # delete all games with playernumber != 10
    for gameNo in list(games.keys()):
        if len(games[gameNo]) != 10:
            del games[gameNo]

    # print number of games
    print("Number of games:", len(games))

    return games


def convert_data(games=None, save_dir="/", filter_matches=CLEAN_MATCHES):
    if games is None:
        games = load_raw_csv()
        games = list(games.values())

    # convert games to tuples
    game_data = []
    print("Converting games...")
    e_count = 0
    for game in tqdm.tqdm(games):
        game_list = convert_game(game, filter_matches=filter_matches)
        if game_list != None:
            game_data.append(game_list)
        else:
            e_count +=1

    print(f"Removed {e_count} games from dataset")

    print("Number of games:", len(game_data))

    # shuffle data
    print("Shuffling data...")
    np.random.shuffle(game_data)

    save_file = os.path.join(save_dir, "game_data_filtered.npy")

    # print length of game data
    print("Length of game data:", len(game_data))
    # save data
    #print("Saving data...")
    game_data = np.array(game_data)
    #np.save(save_file, game_data)
    #print("Done!")

    # save csv
    #print("Saving csv...")
    #save_csv(game_data)
    #print("Done!")

    return game_data


def get_data():
    # downloads and unzips data if not already done
    # check if file exists
    filename = os.path.join(TARGET_DIR, FILENAME)
    if os.path.exists(filename):
        print("File already downloaded!")
        return True
    
    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)

    if not os.path.exists(TARGET_NAME):
        # automatic download doesnt work, so just print url
        print("File not found. Download the dataset and try again.")
        print(URL)
        return False
        #print("Downloading...")
        #urllib.request.urlretrieve(URL, TARGET_NAME)
        #print("Done!")

    print("Unzipping...")
    with zipfile.ZipFile(TARGET_NAME, "r") as zip_ref:
        zip_ref.extractall(TARGET_DIR)

    return True

    print("Done!")

    
def notebook_data():
    games_dict = load_raw_csv()
    games = list(games_dict.values())

    np.random.shuffle(games)

    # split into train, val, test
    train, val, test = split_iterable(games, weights=(90, 5, 5))
    print("train: ", len(train))
    print("val: ", len(val))
    print("test: ", len(test))
    print()

    # convert each match into a list of 10 champions and a 1/0 for win/loss of blue team
    # two copies of train data, one with some matches filtered out
    train, train_filtered = convert_data(train, filter_matches=False), convert_data(train, filter_matches=True)
    val = convert_data(val, filter_matches=False)
    test = convert_data(test, filter_matches=False)

    # save data
    np.save("train.npy", train)
    np.save("train_filtered.npy", train_filtered)
    np.save("val.npy", val)
    np.save("test.npy", test)




if __name__ == "__main__":
    notebook_data()
    exit()
    if get_data():
        convert_data()