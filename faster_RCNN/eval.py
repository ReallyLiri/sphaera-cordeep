import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from config.test_config import test_cfg
from dataloader.sphaera_dataset import sphaera
from utils.draw_box_utils import draw_box
from utils.train_utils import create_model, write_tb
from utils.evaluate_utils import evaluate
from utils.im_utils import Compose, ToTensor, RandomHorizontalFlip

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes=test_cfg.num_classes)

    if torch.cuda.is_available():
        model.cuda()
    weights = test_cfg.model_weights

    checkpoint = torch.load(weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # read class_indict
    data_transform = Compose([ToTensor()])

    test_data_set = sphaera(test_cfg.data_root_dir, 'test', '2017', data_transform, include_class=[0,1,2,3,4])
    batch_size = test_cfg.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(nw))
    test_data_loader = torch.utils.data.DataLoader(test_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=nw,
                                                    collate_fn=test_data_set.collate_fn)
    category_index = test_data_set.class_to_coco_cat_id

    index_category = dict(zip(category_index.values(), category_index.keys()))


    val_mAP = []

    print("------>Starting test data valid")
    _, mAP = evaluate(model, test_data_loader, device=device, mAP_list=val_mAP, extra_val=True, single_cls=test_cfg.single_cls)
    print('test mAp is {}'.format(mAP))

    # board_info = {'val_mAP': mAP}

    # write_tb(writer, epoch, board_info)

    # writer.close()


if __name__ == "__main__":
    version = torch.version.__version__[:5]
    print('torch version is {}'.format(version))
    os.environ["CUDA_VISIBLE_DEVICES"] = test_cfg.gpu_id
    test()
