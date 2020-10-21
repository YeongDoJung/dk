python3 main.py \
	--json_path "/media/disk/dk/configs/2class_mass_dense201.json" \
	--input_path '/media/disk/dk/data/eval_datasets/' \
       	--model_path '/media/disk/dk/experiments/densenet/densenet201_mass/checkpoint/checkpoint.pth.tar' \
       	--output_path '/media/disk/dk/experiments/densenet/densenet201_mass/visual/' \
	--generate_num 100

python3 main.py \
	--json_path "/media/disk/dk/configs/2class_penu_dense201.json" \
	--input_path '/media/disk/dk/data/eval_datasets/' \
       	--model_path '/media/disk/dk/experiments/densenet/densenet201_penu/checkpoint/checkpoint.pth.tar' \
       	--output_path '/media/disk/dk/experiments/densenet/densenet201_penu/visual/' \
	--generate_num 100


python3 main.py \
	--json_path "/media/disk/dk/configs/4class_labels_dense201.json" \
	--input_path '/media/disk/dk/data/eval_datasets/' \
      	--model_path '/media/disk/dk/experiments/densenet/densenet201_4class/checkpoint/checkpoint.pth.tar' \
      	--output_path '/media/disk/dk/experiments/densenet/densenet201_4class/visual/' \
	--generate_num 100


