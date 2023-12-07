# simple CLI to run lol match prediction


# modes:
# 1: enter champion in lobby one by one, with option to predict winrate or best next champion after each entry

class LolPredCLI:

    CHAMP_SELECT_ORDER = [0, 3, 4, 7, 8, 1, 2, 5, 6, 9]

    def __init__(self, model=("SomeClass", "pathToWeights")):

        self.model_path = model[1]
        self.model_class = model[0]

    
        
    def run():
        try:
            pass
        except KeyboardInterrupt:
            print("Exiting...")
            return
        except Exception as e:
            print(e)
            return


if __name__ == "__main__":
    cli = LolPredCLI()
    cli.run()