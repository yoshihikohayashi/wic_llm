#
import tinydb

def main(in_fname='./wic_zspr_gpt-4-0613_test_1.json'):
    db = tinydb.TinyDB(in_fname) 
    tbl_name = list(db.tables())[0]
    table = db.table(tbl_name)    #
    #
    out_fname = in_fname[:-4] + 'tsv'
    with open(out_fname, 'w', encoding='utf-8') as outf:
        m = 0
        for i, inst in enumerate(table.all()): 
            if i==0: outf.write('\t'.join(inst.keys()) + '\n')
            inst_ = [str(_) for _ in inst.values()]
            outf.write('\t'.join(inst_) + '\n')
            #
            if inst['l']==inst['pred']: m += 1
        all = i + 1
    return (m, all)

#####
import sys

if __name__ == '__main__':
    main(in_fname=sys.argv[1])

    
    