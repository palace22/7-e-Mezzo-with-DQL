from typing import Dict, Tuple
from matplotlib import pyplot
from SetteEMezzoDQL import SetteEMezzoDQN
from SetteEMezzoGame import SetteEMezzo
from collections import defaultdict
import numpy as np
from SetteEMezzoMC import SetteEMezzoMC
from SetteEMezzoQL import SetteEMezzoQL


def save_csv(value, file_name="csvfile"):
    with open("csv/" + file_name + ".csv", "w") as file:
        point = "X,"
        for i in np.arange(0, 105.0, 5):
            point += str(i)
            point += ","
        file.write(point)
        file.write("\n")
        pre_row = ""
        row = ""
        for index, v in enumerate(list(value.values())):
            pre_row = str(list(value.keys())[index][0]) + ","
            if v[0] == v[1]:
                row += "?,"
            else:
                row += str(max(v, key=v.get)) + ","
            if list(value.keys())[index][1] == 100:
                file.write(pre_row + row)
                file.write("\n")
                pre_row = ""
                row = ""


def save_csv_mc(value, file_name="csvfile"):
    with open("csv/" + file_name + ".csv", "w") as file:
        point = "X,"
        for i in np.arange(0, 105.0, 5):
            point += str(i)
            point += ","
        file.write(point)
        file.write("\n")
        pre_row = ""
        row = ""
        for index, v in enumerate(list(value.values())):
            pre_row = str(list(value.keys())[index][0]) + ","
            row += str(v) + ","
            if list(value.keys())[index][1] == 100:
                file.write(pre_row + row)
                file.write("\n")
                pre_row = ""
                row = ""


if __name__ == "__main__":
    n_episodes = 1000000
    # sette_e_mezzo = SetteEMezzoMC(n_episodes, policy=(3, 25))
    # sette_e_mezzo = SetteEMezzoQL(n_episodes, eps_start=0.6, policy=(3.5, 25))
    sette_e_mezzo = SetteEMezzoDQN(n_episodes, eps_start=0.4, lr=0.0001)

    v, w, wr = sette_e_mezzo.play()

    save_csv(
        sette_e_mezzo.get_q_table(),
        sette_e_mezzo.__class__.__name__,
    )

    # sette_e_mezzo.save_policy("QLpol")
    # sette_e_mezzo.load_policy("policy_netDQN483.pt", "target_netDQN483.pt")

    v, w, wr = sette_e_mezzo.evaluate()

    #  #### Monte Carlo Every Visit evaluation for every policies ####
    # a = {}
    # max_policy = (-1, -1)
    # max_value = 0

    # for cards_value in np.arange(0.5, 8.0, 0.5):
    #     for bust_prob in range(0, 105, 5):
    #         print(str(cards_value) + "-" + str(bust_prob))
    #         sette_e_mezzo = SetteEMezzoMC(n_episodes, policy=(cards_value, bust_prob))
    #         v, w, wr = sette_e_mezzo.play()
    #         w = wr[len(wr) - 1]
    #         a[(cards_value, bust_prob)] = w
    #         if w > max_value:
    #             max_value = w
    #             max_policy = (cards_value, bust_prob)
    #         save_csv(
    #             sette_e_mezzo.get_q_table(),
    #             sette_e_mezzo.__class__.__name__
    #             + str(cards_value)
    #             + "-"
    #             + str(bust_prob),
    #         )
    #         print(wr[len(wr) - 1])
    #         print(a)
    # save_csv_mc(a, "test")
    # print(a)
    # print(max_policy)
    # print(max_value)
