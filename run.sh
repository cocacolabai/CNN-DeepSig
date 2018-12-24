#!/bin/sh
i=0
while [ $i -lt 20 ]; do
	echo $i
	python3 model.py
	python3 test_saved_model.py $i
	i=$((i+1))
done

python -c "from helper import avg_acu; avg_acu('avg_acu.txt')"
