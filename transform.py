from collections import defaultdict
import random
import numpy as np
import time
import csv
from tqdm import tqdm
import argparse

allslices = []

class CFG(object):
    def __init__(self):
        self.prod = defaultdict(list)

    def add_prod(self, lhs, rhs):
        prods = rhs.split(' \\ ')
        allslices.append(lhs)
        for prod in prods:
            for sp in prod.split():
                if sp not in allslices:
                    allslices.append(sp)
        for prod in prods:
            self.prod[lhs].append(tuple(prod.split()))

    def xss_add_prod(self, lhs, rhs):
        """ Add production to the grammar. """
        prods = rhs.split(' | ')
        allslices.append(lhs)
        for prod in prods:
            for sp in prod.split():
                if sp not in allslices:
                    allslices.append(sp)
        for prod in prods:
            self.prod[lhs].append(tuple(prod.split()))

    def get_sli_number(self, choice, slistr):
        for i in range(len(choice)):
            if choice[i] == slistr:
                return i
        return len(choice)

    def get_ch_sli(self, slinum, choice):
        if slinum >= len(choice):
            return 0
        else:
            return choice[slinum]

def main(grammar_path, data_path):
    cfg = CFG()
    # Đọc grammar từ file
    with open(grammar_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            bnflist = line.split(':=')
            if "xss" in grammar_path:
                cfg.xss_add_prod(bnflist[0], bnflist[1])
            else:
                cfg.add_prod(bnflist[0], bnflist[1])

    # Loại bỏ các phần tử trùng trong allslices
    newallsli = []
    for slic in allslices:
        if slic not in newallsli:
            newallsli.append(slic)

    # Đọc dữ liệu từ file CSV
    with open(data_path, 'r') as f:
        choicelist = [i[0].strip().split(' ') for i in csv.reader(f)]
        choicelist = choicelist[1:]  # Bỏ dòng đầu tiên (header)

    # Chuyển đổi các giá trị trong choicelist thành số nguyên
    for i in range(len(choicelist)):
        for j in range(len(choicelist[i])):
            choicelist[i][j] = int(choicelist[i][j])

    listss = []
    # Tạo dữ liệu mới từ các lựa chọn
    for j, schoice in enumerate(tqdm(choicelist)):
        global datafram
        datafram = schoice
        tmpstr = ''
        for dnum in datafram:
            sli = cfg.get_ch_sli(int(dnum), choice=newallsli)
            if sli not in cfg.prod:
                tmpstr = tmpstr + sli
        listss.append(tmpstr)

    # Lưu kết quả vào file CSV
    with open("output_transformed_ragfast.csv", "w", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["Generated Data"])  # Header
        for row in listss:
            writer.writerow([row])  # Lưu từng dòng đã chuyển đổi vào file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform data based on CFG grammar")
    parser.add_argument('--grammar_path', required=True, help="Path to the grammar file")
    parser.add_argument('--data_path', required=True, help="Path to the data file (CSV)")
    args = parser.parse_args()
    main(args.grammar_path, args.data_path)
