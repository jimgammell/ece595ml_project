for p in $(seq 0.0 0.05 1.0); do
	echo "Beginning experiment with p=$p...";
	python -m cifar.cifar_train --config cifar/configs/cifar-resnet-32.prototxt --noise_ratio $p --finetune >> puf_log__ft;
done
