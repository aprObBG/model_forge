for i in `seq 1 10 49`
do
 python main.py --train-freeze --seed ${i} --proportion 0.2 --admode 'training' --gpu 0 | tee ${i}.log
done
