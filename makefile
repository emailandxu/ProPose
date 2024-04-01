run:
	# python -m debugpy --wait-for-client --listen 5678 \
	python \
	scripts/demo.py \
	--img-dir ./testing_data/daiqin/images \
	--out-dir ./testing_data/daiqin/output \
	--ckpt './model_files/propose_hr48_xyz.pth'
clean:
	rm -rf dump_demo