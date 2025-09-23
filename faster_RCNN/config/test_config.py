class Config:
    model_weights = "/home/jmartinetz/faster_rcnn/runs/1280_flip_SVEDX/resnet50_fpn-model-22-mAp-0.8009364129768686.pth"
    # model_weights = "/home/jmartinetz/faster_rcnn/runs/1280_flip_SVED4000/resnet50_fpn-model-67-mAp-0.6706628222126685.pth"
    # model_weights = "/home/jmartinetz/faster_rcnn/runs/640_flip_SVEDX/resnet50_fpn-model-28-mAp-0.754348001520792.pth"
    image_path = "../data/regions640/images/test/permanent!sphaera!1659_sacrobosco_sphera_1518!pageimg!1659_sacrobosco_sphera_1518_p135.jpg"
    gpu_id = '0'
    num_classes = 5 + 1
    data_root_dir = "../data/regions1280"
    batch_size = 8
    single_cls = True

test_cfg = Config()
