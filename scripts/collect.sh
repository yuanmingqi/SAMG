# for num_obj in 1 2 3;
# do
# 	python scripts/collect.py --tag high --num_episode 100 --seed=${seed} > datasets/output_${seed}.log 2>&1 &
# done

# wait

for num_obj in 1 2 3 4 5 6 7 8 9 10;
do
	python scripts/collect_single.py --num_obj=${num_obj};
done