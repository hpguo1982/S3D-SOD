# coding=utf-8
import os

from skimage import io
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
#from model.EDRNet import EDRNet
import glob
import timeit

from model import SSDSeg

args = {
	'Scale': 256,
    'workers': 2,
    'tst_img_dir': '../data/SD-saliency-900/images/',              # path of training images
    'tst_lbl_dir': '../data/SD-saliency-900/annotations/',             # path of training labels
	'prd_dir': '../data/SD-saliency-900/evaluation/pred_maps/',
    'image_ext': '.bmp',
    'label_ext': '.png',
    'checkpoint': './SSDSeg.pth',
	'source_data': 'validation'
}


Net = SSDSeg
device = 'cuda:0'


def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn


def save_output(image_name, pred, d_dir):
	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()
	im = Image.fromarray(predict_np*255).convert('RGB')
	image = io.imread(image_name)
	imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
	img_name = image_name.split("/")[-1]
	imidx = img_name.split(".")[0]
	imo.save(d_dir+imidx+'.png')


def main():

	# --------- 1. get image path and name ---------
	image_dir = args['tst_img_dir'] + args['source_data'] + '/'              # path of testing dataset
	prediction_dir = args['prd_dir'] +  args['source_data'] + '/'            # path of saving results
	model_dir = args['checkpoint']    # path of pre-trained model
	img_name_list = glob.glob(image_dir + '*.bmp')
	if not os.path.exists(prediction_dir):
		os.makedirs(prediction_dir)
		print(f"目录 {prediction_dir} 已创建。")
	else:
		print(f"目录 {prediction_dir} 已存在。")

	# --------- 2. dataloader ---------
	test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
						transform=transforms.Compose([RescaleT(args['Scale']), ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset,
						batch_size=1, shuffle=False, num_workers=args['workers'])

	# --------- 3. model define ---------
	print("...load PVTSeg from " + model_dir)
	net = Net()
	dict1 = net.state_dict()
	#print(net.state_dict())
	#dict2 =
	net.load_state_dict(torch.load(model_dir))
	net.cuda(device=device)
	net.eval()

	start = timeit.default_timer()
	# --------- 4. inference for each image ---------
	with torch.no_grad():
		for i_test, data_test in enumerate(test_salobj_dataloader):
			print("inferencing:", img_name_list[i_test].split("/")[-1])
			inputs_test = data_test['image']
			inputs_test = inputs_test.type(torch.FloatTensor)
			inputs_test = inputs_test.cuda(device=device)

			s_out = net(inputs_test)
			# normalization
			pred = s_out[:, 0, :, :]
			pred = torch.sigmoid(pred)
			pred = normPRED(pred)

			# save results to test_results folder
			save_output(img_name_list[i_test], pred, prediction_dir)

	end = timeit.default_timer()
	print(str(end-start))




if __name__ == "__main__":
	main()

