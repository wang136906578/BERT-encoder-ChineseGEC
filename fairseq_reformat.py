import sys, os

file_name = sys.argv[1]
input_file = open(file_name, "r").readlines()
sent_list = [None]*2000
with open('./' + os.path.basename(file_name) + '.reformat', "w") as f:
    for line in input_file:
        if line[0] == "H":
            line_split = line.split("\t")
            idx = int(line_split[0][2:])
            sentence = line_split[2]
            sent_list[idx] = sentence
    for i in sent_list:
        print(i.strip(), file=f)
