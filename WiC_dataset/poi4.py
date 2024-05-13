#
fr = 1100; to = 1149
fout_name = './test/test.gold4_' + str(fr) + '_' + str(to) + '.txt'
fout = open(fout_name, 'w')
fin = open('./test/test.gold.txt', 'r')
#
for line in fin.readlines()[fr:to+1]:
    for i in range(4):
        print(line)
        fout.write(line)
fout.close()
