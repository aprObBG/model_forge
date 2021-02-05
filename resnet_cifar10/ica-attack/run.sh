for i in `seq 1 10 49`
do
 python main.py --train-freeze --seed ${i} --proportion 0.025 --admode 'testing' | tee ${i}.log
done
