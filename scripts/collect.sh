for seed in 1 2 3 4 5 6 7 8 9 0;
do
	python scripts/collect.py --tag high --num_episode 200 --seed=${seed} > datasets/output_${seed}.log 2>&1 &
done

wait
