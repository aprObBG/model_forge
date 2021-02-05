for i in `seq 1 10 49`
do
 python main.py --train-freeze --seed ${i} --proportion 0.2 --admode 'training' | tee ${i}.log
 #python main.py --train-freeze --seed ${i} --proportion 0.025 --admode 'testing' | tee ${i}.log
 #python main.py --train-freeze --seed ${i} --proportion 0.25 --admode 'training'
 #python main.py --seed ${i} --proportion 0 --admode 'training'
done

# for i in `seq 10 10 19`
# do
#  python main.py --train-freeze --seed ${i} --proportion 0.025 --admode 'testing' | tee ${i}.log
#  #python main.py --train-freeze --seed ${i} --proportion 0.25 --admode 'training'
#  #python main.py --seed ${i} --proportion 0 --admode 'training'
# done
