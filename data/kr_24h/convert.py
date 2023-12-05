

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


CLEAN_MATCHES

def filter_player(player) -> bool:
    """
    Returns true if a player data is above threshold and match should be removed from list
    """

    return False

def convert_game(game)->list[int]:
    # converts a game to a list of ints
    # first 5 are blue champ ids, next 5 are red champ ids, last one is 1 if blue won, 0 if red won
    blue_team = []
    red_team = []
    blue_win = None
    for player in game:
        champion_name = player["championName"]
        champion_id = -1
        while champion_id == -1:
            try:
                champion_id = converter.get_champion_id_from_name(champion_name)
            except:
                champion_name = converter.get_closest_champion_name(champion_name)
                if champion_name is None:
                    raise Exception("Champion name not found!")
        if player["teamId"] == "100":
            blue_team.append(champion_id)
            if blue_win is None:
                blue_win = player["win"] == "True"
            else:
                if blue_win != (player["win"] == "True"):
                    raise Exception("Blue win is not consistent!")
        else:
            red_team.append(champion_id)
            if blue_win is None:
                blue_win = player["win"] == "False"
            else:
                if blue_win == (player["win"] == "True"):
                    raise Exception("Blue win is not consistent!")
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


def convert_data():
    games = load_raw_csv()

    # convert games to tuples
    game_data = []
    print("Converting games...")
    e_count = 0
    for game in tqdm.tqdm(games.values()):
        try:
            game_list = convert_game(game)
            game_data.append(game_list)
        except:
            e_count += 1

    print(f"Removed {e_count} games from dataset")

    print("Number of games:", len(game_data))

    # shuffle data
    print("Shuffling data...")
    np.random.shuffle(game_data)


    # print length of game data
    print("Length of game data:", len(game_data))
    # save data
    print("Saving data...")
    game_data = np.array(game_data)
    np.save("game_data.npy", game_data)
    print("Done!")

    print("Saving csv...")
    save_csv(game_data)
    print("Done!")

    # load data
    print("Loading data...")
    data_loaded = np.load("game_data.npy")
    while input("Do you want to print a random game? (y/n)") == "y":
        print(data_loaded[np.random.randint(0, len(data_loaded))])


def get_data():
    # downloads and unzips data if not already done
    # check if file exists
    filename = os.path.join(TARGET_DIR, FILENAME)
    if os.path.exists(filename):
        print("File already downloaded!")
        return
    
    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)

    if not os.path.exists(TARGET_NAME):
        print("Downloading...")
        urllib.request.urlretrieve(URL, TARGET_NAME)
        print("Done!")

    print("Unzipping...")
    with zipfile.ZipFile(TARGET_NAME, "r") as zip_ref:
        zip_ref.extractall(TARGET_DIR)

    print("Done!")

    


if __name__ == "__main__":
    get_data()
    convert_data()