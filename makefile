videoname?=default
datahome?=./testing_data
python=/home/tony/miniconda3/envs/propose/bin/python
run:to_frames
	# $(python) -m debugpy --wait-for-client --listen 5678 \
	$(python) \
	main.py \
	--img-dir $(datahome)/$(videoname)/images \
	--out-dir $(datahome)/$(videoname)/output \
	--ckpt './model_files/propose_hr48_xyz.pth' \
	--draw
	$(python) postprocess.py $(videoname) $(datahome)/$(videoname)/output $(datahome)/output/
to_frames:
	mkdir -p $(datahome)/$(videoname)/images
	ffmpeg -i $(datahome)/$(videoname)/$(videoname).mp4 $(datahome)/$(videoname)/images/$(videoname)_%08d.png