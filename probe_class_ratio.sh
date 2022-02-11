for p in $(seq 0.9 0.005 0.995); do
	echo "Beginning experiment with p=$p...";
	python -m mnist.mnist_train --pos_ratio $p --nrun 1 --verbose --exp ours;
done
