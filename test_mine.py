from model.mine.ConditionedMine import MINEConditionalNet


if __name__ == "__main__":
    mine = MINEConditionalNet(2, 64)
    print(mine)
    print("Success")