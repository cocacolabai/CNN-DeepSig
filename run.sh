#!/bin/sh

# preprocessing these testing data
# strip their newline character in single sequence
python -c "from helper import strip_newline; strip_newline('dataset/test_NC.fasta')"
python -c "from helper import strip_newline; strip_newline('dataset/test_TM.fasta')"
python -c "from helper import strip_newline; strip_newline('dataset/test_SP.fasta')"

i=0
while [ $i -lt 20 ]; do
	echo $i
	python3 model.py
	python3 test_saved_model.py $i
	i=$((i+1))
done

# This line can be temporarily removed
python -c "from helper import avg_acu; avg_acu('avg_acu.txt')"
