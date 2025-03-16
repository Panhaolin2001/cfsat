import csv

class Syner_Action_rl:

    def __init__(self, csv_path) -> None:
        
        self.csv_path = csv_path

        self.syner_pairlist_dict = self._get_file_syner_pair_coreset()

    def _get_file_syner_pair_coreset(self):
        result_dict = {}
        with open(self.csv_path, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                key = int(row[0])
                value = eval(row[1])
                result_dict[key] = value
        return result_dict

    def get_next_syner_pair_list(self, action_name):
        next_pair_list = []
        for action_sub_pair in self.syner_action_pair:
            if action_name == action_sub_pair[0][0]:
                next_pair_list.append(action_sub_pair)
        
        return next_pair_list
