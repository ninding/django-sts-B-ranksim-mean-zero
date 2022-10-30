tasks = ['./djangoSts/origin/result/','./djangoSts/mean/result/','./djangoSts/zero/result/','./djangoSts/mean_zero/result/']
models = ['ranksim/','ranksim-loss-lds/','ranksim-lds-inv/']
model_name = 'model_state_best.th'
commands = [{"regularization_weight":3e-4,"interpolation_lambda":2,"evaluate":True,"eval_model":''},
{"loss":"focal_l1","regularization_weight":3e-4,"interpolation_lambda":2,"lds":True,"lds_kernel":"gaussian","lds_ks":5,"lds_sigma":2,"evaluate":True,"eval_model":''},
{"reweight":"inverse","regularization_weight":3e-4,"interpolation_lambda":2,"lds":True,"lds_kernel":"gaussian","lds_ks":5,"lds_sigma":2,"evaluate":True,"eval_model":''} ]