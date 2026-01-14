import argparse
import matplotlib.pyplot as plt
import sqlite3
import sys


parser = argparse.ArgumentParser()
parser.add_argument("-o", required=False, action="store_true",help="use to destroy the database")
parser.add_argument("-r", required=False, action="store_true",help="use to plot the second database")
args = parser.parse_args()
if(args.o==1):
    if(args.r==1):
        con = sqlite3.connect("./Mapping2FaultsInj.sql")
    else:
        con = sqlite3.connect("./Mapping1FaultsInj.sql")
    cur = con.cursor()
    cur.execute("delete from FaultInjResults; ")
    con.commit()
    con.close()
else:

    if(args.r==1):
        con = sqlite3.connect("./Mapping2FaultsInj.sql")
    else:
        con = sqlite3.connect("./Mapping1FaultsInj.sql")
    cur = con.cursor()
    res=cur.execute(f"select FunctionalBlock,PacketsDropped,accuracy from FaultInjResults")
    rows=res.fetchall();
    IDs=[]
    Accuracies=[]
    WrongSpikes=[]
    if(len(rows)==0):
        print("Empty database")
        sys.exit()
    for row in rows:
        IDs.append(row[0])
        WrongSpikes.append(row[1])
        Accuracies.append(row[2])
    plt.figure() 
    max_width = 0.5
    auto_width = min(max_width, 0.8 / len(IDs))
    plt.bar(IDs, Accuracies,width=auto_width)
    plt.ylabel('Missed prediction [%]')
    plt.ylim(0,100)
    plt.xticks( rotation=90)
    plt.savefig("./MissedPrediction.pdf", bbox_inches="tight")
    plt.figure() 
    max_width = 0.5
    auto_width = min(max_width, 0.8 / len(IDs))
    plt.bar(IDs, WrongSpikes,width= 0.8 / len(IDs))
    plt.ylim(0,100)
    plt.xticks( rotation=90)
    plt.ylabel('Wrong Spikes [%]')
    plt.savefig("./Wrong Spikes.pdf", bbox_inches="tight")
    con.close()