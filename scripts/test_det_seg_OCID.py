import argparse
import time
import os
import numpy as np
import scipy
import cv2
from functools import partial

import torch
import torch.optim as optim
import torch.utils.data as data
from torch import distributed

import grasp_det_seg.models as models
from grasp_det_seg.algos.detection import PredictionGenerator, ProposalMatcher, DetectionLoss
from grasp_det_seg.algos.fpn import DetectionAlgoFPN, RPNAlgoFPN
from grasp_det_seg.algos.rpn import AnchorMatcher, ProposalGenerator, RPNLoss
from grasp_det_seg.algos.semantic_seg import SemanticSegAlgo, SemanticSegLoss
from grasp_det_seg.config import load_config
from grasp_det_seg.data_OCID import iss_collate_fn, OCIDTestDataset, OCIDTestTransform
from grasp_det_seg.data_OCID.OCID_class_dict import colors_list, cls_list
from grasp_det_seg.data_OCID.sampler import DistributedARBatchSampler
from grasp_det_seg.models.det_seg import DetSegNet, NETWORK_INPUTS
from grasp_det_seg.modules.fpn import FPN, FPNBody
from grasp_det_seg.modules.heads import RPNHead, FPNROIHead, FPNSemanticHeadDeeplab
from grasp_det_seg.utils import logging
from grasp_det_seg.utils.meters import AverageMeter
from grasp_det_seg.utils.misc import config_to_string, scheduler_from_config, norm_act_from_config, freeze_params, \
    all_reduce_losses, NORM_LAYERS, OTHER_LAYERS
from grasp_det_seg.utils.parallel import DistributedDataParallel
from grasp_det_seg.utils.snapshot import resume_from_snapshot

parser = argparse.ArgumentParser(description="OCID detection and segmentation test script")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--log_dir", type=str, default=".", help="Write logs to the given directory")
parser.add_argument("config", metavar="FILE", type=str, help="Path to configuration file")
parser.add_argument("model", metavar="FILE", type=str, help="Path to model file")
parser.add_argument("data", metavar="DIR", type=str, help="Path to dataset")
parser.add_argument("out_dir", metavar="DIR", type=str, help="Path to output directory")


def save_param_file(writer, param_file):
    data_sum = ''
    with open(param_file) as fp:
        Lines = fp.readlines()
        for line in Lines:
            data_sum += line + '  \n'
    writer.add_text('dataset_parameters', data_sum)
    return


def ensure_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass


def Rotate2D(pts, cnt, ang):
    ang = np.deg2rad(ang)
    return scipy.dot(pts - cnt, scipy.array([[scipy.cos(ang), scipy.sin(ang)], [-scipy.sin(ang),
                                                                                scipy.cos(ang)]])) + cnt


def save_prediction_image(raw_pred, img_abs_path, img_root_path, im_size, out_dir):
    num_classes_theta = 18
    threshold = 0.05
    iou_seg_threshold = 100  # in px

    for i, (sem_pred, bbx_pred, cls_pred, obj_pred) in enumerate(zip(
            raw_pred["sem_pred"], raw_pred["bbx_pred"], raw_pred["cls_pred"], raw_pred["obj_pred"])):

        item = os.path.join(img_root_path[i], img_abs_path[i])
        im_size_ = im_size[i]
        ensure_dir(out_dir)

        seq_path, im_name = item.split(',')
        sem_pred = np.asarray(sem_pred.detach().cpu().numpy(), dtype=np.uint8)
        seg_mask_vis = np.zeros((im_size_[0], im_size_[1], 3))
        cls_labels = np.unique(sem_pred)

        img_path = os.path.join(img_root_path[i], seq_path, 'rgb', im_name)
        mask_path = os.path.join(img_root_path[i], seq_path, 'seg_mask_labeled_combi', im_name)
        img = cv2.imread(img_path)
        img_best_boxes = np.copy(img)
        mask_gt = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        for cnt, label in enumerate(cls_labels):
            if label == 0:
                continue

            seg_mask_vis[sem_pred == label] = colors_list[label]
            mask_per_label = np.zeros_like(sem_pred)
            mask_per_label_gt = np.zeros_like(sem_pred)

            mask_per_label[sem_pred == label] = 1
            mask_per_label_gt[mask_gt == label] = 1

            if sum(map(sum, mask_per_label)) < iou_seg_threshold:
                continue

        ensure_dir(out_dir)
        out_path = os.path.join(out_dir, im_name[:-4] + ".png")

        img_mask = (img * 0.25 + seg_mask_vis * 0.75)

        if bbx_pred is None:
            continue

        anno_per_class_dir = os.path.join(os.path.join(img_root_path[i], seq_path, 'Annotations_per_class',
                                                       im_name[:-4]))

        for class_dir in os.listdir(anno_per_class_dir):
            if not os.path.isdir(os.path.join(anno_per_class_dir, class_dir)):
                continue

            best_confidence = 0.
            r_bbox_best = None

            for bbx_pred_i, cls_pred_i, obj_pred_i in zip(bbx_pred, cls_pred, obj_pred):
                if obj_pred_i.item() > threshold:

                    pt1 = (int(bbx_pred_i[0]), int(bbx_pred_i[1]))
                    pt2 = (int(bbx_pred_i[2]), int(bbx_pred_i[3]))
                    cls = cls_pred_i.item()
                    if cls > 17:
                        assert False

                    theta = ((180 / num_classes_theta) * cls) + 5
                    pts = scipy.array([[pt1[0], pt1[1]], [pt2[0], pt1[1]], [pt2[0], pt2[1]], [pt1[0], pt2[1]]])
                    cnt = scipy.array([(int(bbx_pred_i[0]) + int(bbx_pred_i[2])) / 2,
                                       (int(bbx_pred_i[1]) + int(bbx_pred_i[3])) / 2])
                    r_bbox_ = Rotate2D(pts, cnt, 90 - theta)
                    r_bbox_ = r_bbox_.astype('int16')

                    if (int(cnt[1]) >= im_size_[0]) or (int(cnt[0]) >= im_size_[1]):
                        continue

                    if sem_pred[int(cnt[1]), int(cnt[0])] == int(class_dir):

                        if obj_pred_i.item() >= best_confidence:
                            best_confidence = obj_pred_i.item()
                            r_bbox_best = r_bbox_

            if r_bbox_best is not None:
                cv2.line(img_best_boxes, tuple(r_bbox_best[0]), tuple(r_bbox_best[1]), (255, 0, 0), 2)
                cv2.line(img_best_boxes, tuple(r_bbox_best[1]), tuple(r_bbox_best[2]), (0, 0, 255), 2)
                cv2.line(img_best_boxes, tuple(r_bbox_best[2]), tuple(r_bbox_best[3]), (255, 0, 0), 2)
                cv2.line(img_best_boxes, tuple(r_bbox_best[3]), tuple(r_bbox_best[0]), (0, 0, 255), 2)

        res = np.hstack((img, img_best_boxes, img_mask))
        scale_percent = 75  # percent of original size
        width = int(res.shape[1] * scale_percent / 100)
        height = int(res.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(res, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_path, resized)
        return


def log_debug(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().info(msg, *args, **kwargs)


def make_config(args):
    log_debug("Loading configuration from %s", args.config)
    conf = load_config(args.config, args.config)
    log_debug("\n%s", config_to_string(conf))
    return conf


def make_dataloader(args, config, rank, world_size):
    config = config["dataloader"]
    log_debug("Creating dataloaders for dataset in %s", args.data)

    # Validation dataloader
    val_tf = OCIDTestTransform(config.getint("shortest_size"),
                               config.getint("longest_max_size"),
                               config.getstruct("rgb_mean"),
                               config.getstruct("rgb_std")
                               )
    val_db = OCIDTestDataset(args.data, config["root_path"], config["test_set"], val_tf)
    val_sampler = DistributedARBatchSampler(
        val_db, config.getint("val_batch_size"), world_size, rank, False)
    val_dl = data.DataLoader(val_db,
                             batch_sampler=val_sampler,
                             collate_fn=iss_collate_fn,
                             pin_memory=True,
                             num_workers=config.getint("num_workers"))
    return val_dl


def make_model(config):
    body_config = config["body"]
    fpn_config = config["fpn"]
    rpn_config = config["rpn"]
    roi_config = config["roi"]
    sem_config = config["sem"]
    general_config = config["general"]
    classes = {"total": int(general_config["num_things"]) + int(general_config["num_stuff"]), "stuff":
        int(general_config["num_stuff"]), "thing": int(general_config["num_things"]),
               "semantic": int(general_config["num_semantic"])}

    # BN + activation
    norm_act_static, norm_act_dynamic = norm_act_from_config(body_config)

    # Create backbone
    log_debug("Creating backbone model %s", body_config["body"])
    body_fn = models.__dict__["net_" + body_config["body"]]
    body_params = body_config.getstruct("body_params") if body_config.get("body_params") else {}
    body = body_fn(norm_act=norm_act_static, **body_params)
    if body_config.get("weights"):
        body.load_state_dict(torch.load(body_config["weights"], map_location="cpu"))

    # Freeze parameters
    for n, m in body.named_modules():
        for mod_id in range(1, body_config.getint("num_frozen") + 1):
            if ("mod%d" % mod_id) in n:
                freeze_params(m)

    body_channels = body_config.getstruct("out_channels")

    # Create FPN
    fpn_inputs = fpn_config.getstruct("inputs")
    fpn = FPN([body_channels[inp] for inp in fpn_inputs],
              fpn_config.getint("out_channels"),
              fpn_config.getint("extra_scales"),
              norm_act_static,
              fpn_config["interpolation"])
    body = FPNBody(body, fpn, fpn_inputs)

    # Create RPN
    proposal_generator = ProposalGenerator(rpn_config.getfloat("nms_threshold"),
                                           rpn_config.getint("num_pre_nms_train"),
                                           rpn_config.getint("num_post_nms_train"),
                                           rpn_config.getint("num_pre_nms_val"),
                                           rpn_config.getint("num_post_nms_val"),
                                           rpn_config.getint("min_size"))
    anchor_matcher = AnchorMatcher(rpn_config.getint("num_samples"),
                                   rpn_config.getfloat("pos_ratio"),
                                   rpn_config.getfloat("pos_threshold"),
                                   rpn_config.getfloat("neg_threshold"),
                                   rpn_config.getfloat("void_threshold"))
    rpn_loss = RPNLoss(rpn_config.getfloat("sigma"))
    rpn_algo = RPNAlgoFPN(
        proposal_generator, anchor_matcher, rpn_loss,
        rpn_config.getint("anchor_scale"), rpn_config.getstruct("anchor_ratios"),
        fpn_config.getstruct("out_strides"), rpn_config.getint("fpn_min_level"), rpn_config.getint("fpn_levels"))
    rpn_head = RPNHead(
        fpn_config.getint("out_channels"), len(rpn_config.getstruct("anchor_ratios")), 1,
        rpn_config.getint("hidden_channels"), norm_act_dynamic)

    # Create detection network
    prediction_generator = PredictionGenerator(roi_config.getfloat("nms_threshold"),
                                               roi_config.getfloat("score_threshold"),
                                               roi_config.getint("max_predictions"))
    proposal_matcher = ProposalMatcher(classes,
                                       roi_config.getint("num_samples"),
                                       roi_config.getfloat("pos_ratio"),
                                       roi_config.getfloat("pos_threshold"),
                                       roi_config.getfloat("neg_threshold_hi"),
                                       roi_config.getfloat("neg_threshold_lo"),
                                       roi_config.getfloat("void_threshold"))
    roi_loss = DetectionLoss(roi_config.getfloat("sigma"))
    roi_size = roi_config.getstruct("roi_size")
    roi_algo = DetectionAlgoFPN(
        prediction_generator, proposal_matcher, roi_loss, classes, roi_config.getstruct("bbx_reg_weights"),
        roi_config.getint("fpn_canonical_scale"), roi_config.getint("fpn_canonical_level"), roi_size,
        roi_config.getint("fpn_min_level"), roi_config.getint("fpn_levels"))
    roi_head = FPNROIHead(fpn_config.getint("out_channels"), classes, roi_size, norm_act=norm_act_dynamic)

    # Create semantic segmentation network
    sem_loss = SemanticSegLoss(ohem=sem_config.getfloat("ohem"))
    sem_algo = SemanticSegAlgo(sem_loss, classes["semantic"])
    sem_head = FPNSemanticHeadDeeplab(fpn_config.getint("out_channels"),
                                      sem_config.getint("fpn_min_level"),
                                      sem_config.getint("fpn_levels"),
                                      classes["semantic"],
                                      pooling_size=sem_config.getstruct("pooling_size"),
                                      norm_act=norm_act_static)

    # Create final network
    return DetSegNet(body, rpn_head, roi_head, sem_head, rpn_algo, roi_algo, sem_algo, classes)


def make_optimizer(config, model, epoch_length):
    body_config = config["body"]
    opt_config = config["optimizer"]
    sch_config = config["scheduler"]

    # Gather parameters from the network
    norm_parameters = []
    other_parameters = []
    for m in model.modules():
        if any(isinstance(m, layer) for layer in NORM_LAYERS):
            norm_parameters += [p for p in m.parameters() if p.requires_grad]
        elif any(isinstance(m, layer) for layer in OTHER_LAYERS):
            other_parameters += [p for p in m.parameters() if p.requires_grad]
    assert len(norm_parameters) + len(other_parameters) == len([p for p in model.parameters() if p.requires_grad]), \
        "Not all parameters that require grad are accounted for in the optimizer"

    # Set-up optimizer hyper-parameters
    parameters = [
        {
            "params": norm_parameters,
            "lr": opt_config.getfloat("lr") if not body_config.getboolean("bn_frozen") else 0.,
            "weight_decay": opt_config.getfloat("weight_decay") if opt_config.getboolean("weight_decay_norm") else 0.
        },
        {
            "params": other_parameters,
            "lr": opt_config.getfloat("lr"),
            "weight_decay": opt_config.getfloat("weight_decay")
        }
    ]

    optimizer = optim.SGD(
        parameters, momentum=opt_config.getfloat("momentum"), nesterov=opt_config.getboolean("nesterov"))

    scheduler = scheduler_from_config(sch_config, optimizer, epoch_length)

    assert sch_config["update_mode"] in ("batch", "epoch")
    batch_update = sch_config["update_mode"] == "batch"
    total_epochs = sch_config.getint("epochs")

    return optimizer, scheduler, batch_update, total_epochs


def test(model, dataloader, **varargs):
    model.eval()
    dataloader.batch_sampler.set_epoch(0)
    data_time_meter = AverageMeter(())
    batch_time_meter = AverageMeter(())
    data_time = time.time()

    for it, batch in enumerate(dataloader):
        print('Batch no. : ' + str(it))
        with torch.no_grad():
            # Extract data
            img = batch["img"].cuda(device=varargs["device"], non_blocking=True)
            abs_paths = batch["abs_path"]
            root_paths = batch["root_path"]
            im_size = batch["im_size"]
            data_time_meter.update(torch.tensor(time.time() - data_time))
            batch_time = time.time()

            # Run network
            _, pred, conf = model(img=img, do_loss=False, do_prediction=True)

            # Update meters
            batch_time_meter.update(torch.tensor(time.time() - batch_time))

            varargs["save_function"](pred, abs_paths, root_paths, im_size)
            data_time = time.time()


def main(args):
    # Initialize multi-processing
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = args.local_rank, torch.device(args.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Load configuration
    config = make_config(args)
    # Create dataloaders
    test_dataloader = make_dataloader(args, config, rank, world_size)

    # Create model
    model = make_model(config)

    log_debug("Loading snapshot from %s", args.model)
    snapshot = resume_from_snapshot(model, args.model, ["body", "rpn_head", "roi_head", "sem_head"])

    # Init GPU stuff
    torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
    model = DistributedDataParallel(model.cuda(device), device_ids=[device_id], output_device=device_id,
                                    find_unused_parameters=True)

    save_function = partial(save_prediction_image, out_dir=args.out_dir)
    test(model, test_dataloader, device=device, summary=None,
         log_interval=config["general"].getint("log_interval"), save_function=save_function)


if __name__ == "__main__":
    main(parser.parse_args())
