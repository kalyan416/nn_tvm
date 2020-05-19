import os
import torch
import torchvision.transforms as transforms
#from resnet50_3d import *
import tvm
from tvm import relay
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import time
import tvm.contrib.graph_runtime as runtime

network = 'resnet-50_1d_f_nano_gpu'
log_file = "%s.log" % network
dtype = 'float32'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 600,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150)),
}
tuning_rpc_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 600,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=1000),
        runner=autotvm.RPCRunner(
            'nano', host='0.0.0.0', port=9190,
            number=2,
            repeat = 3,
            timeout=2000,
            #min_repeat_ms=150
        ),),
}
def prune_old_tasks(tasks, log_file):
    tmp_log_file = log_file + ".tmp"
    if os.path.isfile(log_file):
    	autotvm.record.pick_best(tmp_log_file, log_file)
    if os.path.isfile(log_file):
        new_tasks = []
        history = autotvm.record.ApplyHistoryBest(log_file)
        for task in tasks:
            if history._query_inside(task.target, task.workload) is None:
                new_tasks.append(task)
        return new_tasks
    else:
        return tasks

import tempfile
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    #if os.path.exists(tmp_log_file):
     #   os.remove(tmp_log_file)
    check_tmp_his = True
    ind = 0
    if check_tmp_his:
    	length = 0
    	if os.path.isfile(tmp_log_file):
    		lines = list(open(tmp_log_file))
    		length = len(lines)
    	else:
    		check_tmp_his = False
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        '''if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))'''

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        print(ind)
        if check_tmp_his:
        	if ind < length:
        		ret = autotvm.record.decode(lines[ind])
        		inp, _ = ret
        		if inp.task.workload == tsk.workload:
        			if (ind + tsk_trial - 1) < length:
        				ind = (ind + tsk_trial - 1)
        				ret_end = autotvm.record.decode(lines[ind])
        				inp_end , _= ret_end
        				if inp.task.workload == inp_end.task.workload:
        					ind = ind + 1
        					continue
        				else:
        					check_tmp_his = False
        					tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
        			elif (ind + tsk_trial - 1) == length:
        				ind = (ind + tsk_trial - 1)
        				ret_end = autotvm.record.decode(lines[ind])
        				inp_end , _= ret_end
        				if inp.task.workload == inp_end.task.workload:
        					ind = ind + 1
        					check_tmp_his = False
        					tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
        					continue
        				else:
        					check_tmp_his = False
        					tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
        			else:
        				cmtd = length - ind
        				ind = length - 1
        				ret_end = autotvm.record.decode(lines[ind])
        				inp_end , _= ret_end
        				if inp.task.workload == inp_end.task.workload:
        					ind = ind + 1
        					check_tmp_his = False
        					tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
        					tsk_trial = tsk_trial - cmtd
        				else:
        					check_tmp_his = False
        					tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
        with tempfile.NamedTemporaryFile() as tmp_task_log_file:
        		tuner_obj.tune(n_trial=tsk_trial,
        				early_stopping=early_stopping,
        				measure_option=measure_option,
        				callbacks=[
        					autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
        					autotvm.callback.log_to_file(tmp_task_log_file.name)])
        		with open(tmp_log_file, 'a') as tmp_log:
        			tmp_log.write(tmp_task_log_file.read().decode('utf8'))

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

calibration_samples = 10
def calibrate_dataset(img_path,b_size=1):

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	
	val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(img_path, transforms.Compose([
							transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize,])),
							batch_size=b_size, shuffle=False,num_workers=1, pin_memory=True)
	
	for i, (batch, _) in enumerate(val_loader):
		if i * b_size >= calibration_samples:
			break
		data = batch.cpu().numpy()
		yield {'input0': data}
		
def quantize(mod, params,img_path, data_aware,b_size=1):
	if data_aware:
		with relay.quantize.qconfig(calibrate_mode='kl_divergence', weight_scale='max'):
			mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset(img_path,1))
	else:
		with relay.quantize.qconfig(calibrate_mode='global_scale', global_scale=8.0):
			mod = relay.quantize.quantize(mod, params)
	return mod
def run_inf(mod,img_path,b_size=1):
	with autotvm.apply_history_best(log_file):
		print("Compile...")
		with relay.build_config(opt_level=3):
			graph, lib, params = relay.build_module.build(mod, target=tvm.target.cuda())

		ctx = tvm.context(str(tvm.target.cuda()), 0)
		module = runtime.create(graph, lib, ctx)
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
		val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(img_path, transforms.Compose([
							transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize,])),
							batch_size=b_size, shuffle=False,num_workers=1, pin_memory=True)
		total = 0
		top1 = 0
		start = time.time()
		total_time = 0
		for i, (batch,target) in enumerate(val_loader):
			data = batch.cpu().numpy()
			total = i
			module.set_input('input0', data)
			module.set_input(**params)
			module.run()
			prediction = module.get_output(0)
			if np.argmax(prediction.asnumpy()[0]) == target.cpu().numpy()[0] :
				top1 = top1+1
				print(top1)
			#if i > 9:  # only run inference on a few samples in this tutorial
			#	break
		end = time.time()
		ftimer = module.module.time_evaluator('run',ctx,1,1000)
		prof_res = np.array(ftimer().results) * 1000 

		print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))
		print('total time for 9 images:{}(sec)'.format(end-start))
		print('total :{} top1 : {} accu: {}'.format(total,top1,top1/float(total)))
###############################################################################
# Run Inference
# -------------
# We create a Relay VM to build and execute the model.
def run_inference(mod,img_path,b_size=1):
	target1 = 'cuda'
	ctx = tvm.gpu(0)
	with autotvm.apply_history_best(log_file):
		print("Compile...")
		executor = relay.create_executor('vm', mod, ctx, target1)
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

	#img = Image.open('/home/kalyan/.tvm_test_data/data/cat.png').resize((224, 224))
	#my_preprocess = transforms.Compose([
	#transforms.Resize(256),
	#transforms.CenterCrop(224),
	#transforms.ToTensor(),normalize])
	#img = my_preprocess(img)
	#img = np.expand_dims(img, 0)
	#prediction = executor.evaluate()(img)
	#top1_tvm = np.argmax(prediction.asnumpy()[0])
	
	with open('/home/kalyan/.tvm_test_data/data/imagenet_synsets.txt') as f:
		synsets = f.readlines()

	synsets = [x.strip() for x in synsets]
	splits = [line.split(' ') for line in synsets]
	key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}
	
	with open('/home/kalyan/.tvm_test_data/data/imagenet_classes.txt') as f:
		class_id_to_key = f.readlines()
	class_id_to_key = [x.strip() for x in class_id_to_key]
	#tvm_class_key = class_id_to_key[top1_tvm]
	#print('Relay top-1 id: {}, class name: {}'.format(top1_tvm, key_to_classname[tvm_class_key]))
	# image is loaded in the PIL RGB format different from caffe-opencv format  
	val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(img_path, transforms.Compose([
							transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize,])),
							batch_size=b_size, shuffle=False,num_workers=1, pin_memory=True)
	total = 0
	top1 = 0
	executor = executor.evaluate()
	start = time.time()
	for i, (batch,target) in enumerate(val_loader):
		data = batch.cpu().numpy()
		total = i
		prediction = executor(data)
		if np.argmax(prediction.asnumpy()[0]) == target.cpu().numpy()[0] :
			top1 = top1+1
			print(top1)
		'''if i > 9:  # only run inference on a few samples in this tutorial
			break'''
	end = time.time()
	print('total time for 9 images:{}(sec)'.format(end-start))
	print('total :{} top1 : {} accu: {}'.format(total,top1,top1/float(total)))
	return
def get_model():
	model = resnet50()
	m_state_dict = torch.load('/home/kalyan/models/pytorch/models/resnet50/resnet50_3d/fused_resnet50.pth')
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in m_state_dict.items():
		name = k[7:] # remove `module.`
		new_state_dict[name] = v
	model.load_state_dict(new_state_dict)
	model = model.eval()
	input_shape = [1, 3, 224, 224]
	input_data = torch.randn(input_shape)
	scripted_model = torch.jit.trace(model, input_data).eval()
	input_name = 'input0'
	shape_list = [(input_name, (1,3,224,224))]
	mod, params = relay.frontend.from_pytorch(scripted_model,shape_list)
	return mod,params

'''def main():
	print("Loading model ...")
	input_shape = (1, 3, 224, 224)
	output_shape = (1, 1000)
	mod, params = get_model()
    
	img_path = '/home/kalyan/Downloads/imageNet/val'
	print('quantizing...')
	mod = quantize(mod, params, img_path ,data_aware=True)
	
	print("Extract tasks...")
	#tasks = autotvm.task.extract_from_program(mod, target=tvm.target.cuda(),params=params,ops=(relay.op.get("nn.conv2d"),))
	# run tuning tasks
	print("Tuning...")
	#tune_tasks(tasks, **tuning_option)
	run_inf(mod,img_path)
	#run_inference(mod,img_path)

if __name__ == '__main__':
	main()'''
