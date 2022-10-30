commands=("python train.py --regularization_weight=3e-4 --interpolation_lambda=2" "python train.py --loss focal_l1 --regularization_weight=3e-4 --interpolation_lambda=2  --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2" "python train.py --reweight inverse --regularization_weight=3e-4 --interpolation_lambda=2 --lds --lds_kernel gaussian --lds_ks 5 --lds_sigma 2" )

for(( i=0;i<${#commands[@]};i++)) do
#${#array[@]}获取数组长度用于循环
echo ${commands[i]} > "./result/${commands[i]}.txt";
${commands[i]}
done;
