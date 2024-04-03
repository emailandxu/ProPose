videoname?=default
run:
	# python -m debugpy --wait-for-client --listen 5678 \
	python \
	main.py \
	--img-dir ./testing_data/$(videoname)/images \
	--out-dir ./testing_data/$(videoname)/output \
	--ckpt './model_files/propose_hr48_xyz.pth'
	python postprocess.py $(videoname) ./testing_data/$(videoname)/output ./testing_data/output/
