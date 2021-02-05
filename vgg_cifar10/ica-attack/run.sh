for i in `seq 1 10 49`
do
 python main.py --train-freeze --seed ${i} --proportion 0.025 --admode 'testing' --gpu 0 | tee ${i}.log
 #python main.py --train-freeze --seed ${i} --proportion 0.75 --admode 'training' --switch 'direct'
 #python main.py --seed ${i} --proportion 0 --admode 'training'
done

# for i in `seq 10 10 19`
# do
#  python main.py --train-freeze --seed ${i} --proportion 0.025 --admode 'testing' --gpu 2 | tee ${i}.log
#  #python main.py --train-freeze --seed ${i} --proportion 0.75 --admode 'training' --switch 'direct'
#  #python main.py --seed ${i} --proportion 0 --admode 'training'
# done
