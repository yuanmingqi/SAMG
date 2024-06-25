for seed in 1 2 3;
do
	python scripts/collect.py --tag high --num_episode 100 --seed=${seed} > datasets/output_${seed}.log 2>&1 &
done

wait
