echo "Grouping rules..."
mkdir -p group
CMD=""
for i in {63,55,13,12,147,16,30,14,197,104,201,193,179,1,170,21,86,89,136,154,75,84,2,108,141,88,94,123,112,127,152,184,132,35,79,98,105,3,118,129,153,185,78,7,196,155,194,29,4,20,5,188,8,18,198,9,11,187,10,17,200,215}
do
        tmp=" ../data/crs/rule${i}.nfa"
	CMD=${CMD}${tmp}
done
python script.py group/group_0.txt ${CMD}

CMD=""
for i in {211,73,24,19,151,156,34,38,46,182,149,181,203,148,116,131,96,95,137,134,106,103,133,128,119,114,110,87,139,92,81,77,121,109,97,85,83,124,93,80,76,122,101,91,125}
do
        tmp=" ../data/crs/rule${i}.nfa"
	CMD=${CMD}${tmp}
done
python script.py group/group_1.txt ${CMD}

CMD=""
for i in {126,99,130,113,107,146,32,178,135,157,90,100,31,28,169,168,167,209,160,192,145,165,195,159,171,56,143,22,33,115,120}
do
        tmp=" ../data/crs/rule${i}.nfa"
	CMD=${CMD}${tmp}
done
python script.py group/group_2.txt ${CMD}

CMD=""
for i in {140,102,117,111,208,205,202,138,210,61,161,23,27,82,164,216,50,62,26,25,69,174,173,175,176,57,64}
do
        tmp=" ../data/crs/rule${i}.nfa"
	CMD=${CMD}${tmp}
done
python script.py group/group_3.txt ${CMD}

CMD=""
for i in {163,58,59,162,66,36,37,204,49,150,180,177,144,183,53,158,166,48,6}
do
        tmp=" ../data/crs/rule${i}.nfa"
	CMD=${CMD}${tmp}
done
python script.py group/group_4.txt ${CMD}

CMD=""
for i in {52,65,39,70,40,54,72,60,51,15,68,67,74}
do
        tmp=" ../data/crs/rule${i}.nfa"
	CMD=${CMD}${tmp}
done
python script.py group/group_5.txt ${CMD}

CMD=""
for i in {71,172,142,207,45,189,186,44,43}
do
        tmp=" ../data/crs/rule${i}.nfa"
	CMD=${CMD}${tmp}
done
python script.py group/group_6.txt ${CMD}

CMD=""
for i in {41,199,213,214,212}
do
        tmp=" ../data/crs/rule${i}.nfa"
	CMD=${CMD}${tmp}
done
python script.py group/group_7.txt ${CMD}

CMD=""
for i in {190,191}
do
        tmp=" ../data/crs/rule${i}.nfa"
	CMD=${CMD}${tmp}
done
python script.py group/group_8.txt ${CMD}

CMD=""
for i in {42,47}
do
        tmp=" ../data/crs/rule${i}.nfa"
	CMD=${CMD}${tmp}
done
python script.py group/group_9.txt ${CMD}

echo "Grouping Done."

echo "Begin Matching ... "

export LD_LIBRARY_PATH=../build/lib/:$LD_LIBRARY_PATH


:<<H
../build/tools/batch  -f group/group_0.txt -i ../data/bing.txt -n 1024 -p -k iNFA
../build/tools/batch  -f group/group_1.txt -i ../data/bing.txt -n 1024 -p -k iNFA
../build/tools/batch  -f group/group_2.txt -i ../data/bing.txt -n 1024 -p -k iNFA
../build/tools/batch  -f group/group_3.txt -i ../data/bing.txt -n 1024 -p -k iNFA
../build/tools/batch  -f group/group_4.txt -i ../data/bing.txt -n 1024 -p -k iNFA
../build/tools/batch  -f group/group_5.txt -i ../data/bing.txt -n 1024 -p -k iNFA

../build/tools/batch  -f group/group_6.txt -i ../data/bing.txt -n 1024 -p -k AS
../build/tools/batch  -f group/group_7.txt -i ../data/bing.txt -n 1024 -p -k AS
../build/tools/batch  -f group/group_8.txt -i ../data/bing.txt -n 1024 -p -k AS
../build/tools/batch  -f group/group_9.txt -i ../data/bing.txt -n 1024 -p -k AS
H

../build/tools/batch  -f group/group_6.txt -i ../data/bing.txt -n 1024 -p -k AD
