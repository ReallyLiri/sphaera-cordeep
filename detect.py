# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, PDFs, etc.

Usage:
    $ python path/to/detect.py \
      --source 0  # webcam
               img.jpg  # image
               vid.mp4  # video
               document.pdf  # PDF (splits into pages)
               path/  # directory
               path/*.jpg  # glob
               'https://youtu.be/Zgi9g1ksQHc'  # YouTube
               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading
import queue
import contextlib
import logging

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout, stderr, and logging"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_level = logging.getLogger().level

        try:
            sys.stdout = devnull
            sys.stderr = devnull
            logging.getLogger().setLevel(logging.ERROR)  # Suppress INFO/DEBUG logs
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            logging.getLogger().setLevel(old_level)


def convert_pdf_page_to_image(pdf_path, page_num, output_dir):
    """Convert a single PDF page to an image"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
        if page_num >= doc.page_count:
            doc.close()
            raise IndexError(f"Page {page_num} does not exist in PDF with {doc.page_count} pages")

        page = doc[page_num]

        # Convert PDF page to image (300 DPI for good quality)
        mat = fitz.Matrix(300/72, 300/72)  # 300 DPI scaling
        pix = page.get_pixmap(matrix=mat)

        # Save image with page number as filename
        image_path = output_dir / f"{page_num + 1}.jpg"
        pix.save(str(image_path))

        doc.close()
        return str(image_path), page_num + 1
    except Exception as e:
        raise Exception(f"Failed to convert page {page_num}: {str(e)}")


def extract_pdf_page(args):
    """Extract a single PDF page to an image file"""
    pdf_path, page_num, temp_dir = args

    try:
        image_path, page_number = convert_pdf_page_to_image(pdf_path, page_num, temp_dir)
        return image_path, page_number, True
    except Exception as e:
        return None, page_num + 1, False


def process_single_image_with_model(image_path, page_number, model, device, imgsz, conf_thres, iou_thres,
                                  max_det, save_dir, save_txt, save_conf, save_crop, classes,
                                  agnostic_nms, augment, visualize, line_thickness, hide_labels,
                                  hide_conf, half, names, stride):
    """Process a single image with an already loaded model"""
    from utils.datasets import LoadImages

    # Load single image
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=model.pt)

    # Initialize crop counter for this image
    crop_counters = {}

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        t2 = time_sync()

        # Inference
        pred = model(im, augment=augment, visualize=False)
        t3 = time_sync()

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):
            p = Path(path)
            filename = f"{page_number}.jpg"
            save_path = str(save_dir / filename)
            txt_path = str(save_dir / 'labels' / str(page_number))

            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
            imc = im0s.copy() if save_crop else im0s
            annotator = Annotator(im0s, line_width=line_thickness, example=str(names))

            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_crop:
                        c = int(cls)
                        class_name = names[c]

                        # Initialize or increment counter for this class
                        if class_name not in crop_counters:
                            crop_counters[class_name] = 0
                        crop_counters[class_name] += 1

                        # Create new filename format: <page>_<class>_<index>.jpg
                        crop_filename = f"{page_number}_{class_name}_{crop_counters[class_name]}.jpg"
                        crops_dir = save_dir / 'crops'
                        crops_dir.mkdir(parents=True, exist_ok=True)
                        save_one_box(xyxy, imc, file=crops_dir / crop_filename, BGR=True)

                    # Add bbox to image
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Save results (always save for PDF processing)
            im0 = annotator.result()
            cv2.imwrite(save_path, im0)


@torch.no_grad()
def single_run(weights, source, imgsz, conf_thres, iou_thres, max_det, device, view_img, save_txt, save_conf,
               save_crop, nosave, classes, agnostic_nms, augment, visualize, update, project, name, exist_ok,
               line_thickness, hide_labels, hide_conf, half, dnn, page_num=None):
    """Run inference on a single image file"""
    return run(weights, source, imgsz, conf_thres, iou_thres, max_det, device, view_img, save_txt, save_conf,
               save_crop, nosave, classes, agnostic_nms, augment, visualize, update, project, name, exist_ok,
               line_thickness, hide_labels, hide_conf, half, dnn, page_num=page_num)


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='mps',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        page_num=None,  # page number for PDF processing
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_pdf = Path(source).suffix.lower() == '.pdf'
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) or is_pdf
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Handle PDF files with concurrent page extraction and sequential inference
    if is_pdf:
        pdf_path = Path(source)
        source_name = pdf_path.stem

        # Set up single base directory for everything
        save_dir = increment_path(Path(project) / source_name, exist_ok=exist_ok)
        temp_images_dir = save_dir / 'temp_images'
        temp_images_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

        # Get total number of pages
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()

        from tqdm import tqdm
        pbar = tqdm(total=total_pages, desc=f"Processing {source_name}", unit="page")

        start_time = time_sync()

        # Load model once for all pages (suppress initialization prints)
        with suppress_stdout_stderr():
            device = select_device(device)
            model = DetectMultiBackend(weights, device=device, dnn=dnn)
            stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
            imgsz = check_img_size(imgsz, s=stride)

            # Half precision setup
            half &= (pt or jit or engine) and device.type != 'cpu' and device.type != 'mps'
            if pt or jit:
                model.model.half() if half else model.model.float()

            # Warmup model
            model.warmup(imgsz=(1, 3, *imgsz), half=half)

        # Producer-consumer pattern: concurrent extraction, sequential processing
        image_queue = queue.Queue(maxsize=10)  # Buffer up to 10 extracted images
        extraction_complete = threading.Event()

        def extraction_worker():
            try:
                for page_num in range(total_pages):
                    try:
                        image_path, page_number = convert_pdf_page_to_image(pdf_path, page_num, temp_images_dir)
                        image_queue.put((image_path, page_number, True))
                    except Exception as e:
                        LOGGER.error(f"Error extracting page {page_num + 1}: {e}")
                        image_queue.put((None, page_num + 1, False))
            finally:
                extraction_complete.set()

        extraction_thread = threading.Thread(target=extraction_worker)
        extraction_thread.start()
        successful = 0
        processed_count = 0

        while processed_count < total_pages:
            try:
                image_path, page_number, extraction_success = image_queue.get(timeout=30)
                processed_count += 1

                if extraction_success and image_path:
                    try:
                        process_single_image_with_model(
                            image_path, page_number, model, device, imgsz, conf_thres, iou_thres,
                            max_det, save_dir, save_txt, save_conf, save_crop, classes,
                            agnostic_nms, augment, visualize, line_thickness, hide_labels,
                            hide_conf, half, names, stride
                        )
                        successful += 1
                        os.remove(image_path)

                    except Exception as e:
                        LOGGER.error(f"Error processing page {page_number}: {e}")

                pbar.update(1)
                image_queue.task_done()

            except queue.Empty:
                if extraction_complete.is_set():
                    break
                LOGGER.warning("Timeout waiting for next page")

        extraction_thread.join()

        pbar.close()

        end_time = time_sync()
        total_time = end_time - start_time
        avg_time_per_page = total_time / total_pages if total_pages > 0 else 0

        import shutil
        shutil.rmtree(temp_images_dir)

        LOGGER.info(f"Successfully processed {successful}/{total_pages} pages from {pdf_path.name}")
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
        LOGGER.info(f"Statistics: Total time: {total_time:.1f}s, Average time per page: {avg_time_per_page:.1f}s")

        return

    # Directories
    # Use source filename (without extension) as base name
    source_name = Path(source).stem if not webcam and not is_url else name
    save_dir = increment_path(Path(project) / source_name, exist_ok=exist_ok)  # increment run

    # Handle exist_ok flag: clear existing contents if exist_ok=True and directory exists
    if exist_ok and save_dir.exists():
        import shutil
        shutil.rmtree(save_dir)

    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu' and device.type != 'mps'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    crop_counters = {}  # Initialize crop counters for regular processing
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image

            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # Use page number for PDF-derived images, otherwise use original filename
            filename = f"{page_num}.jpg" if page_num else p.name
            save_path = str(save_dir / filename)  # im.jpg
            txt_path = str(save_dir / 'labels' / (page_num if page_num else p.stem)) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            class_name = names[c]

                            # Initialize or increment counter for this class
                            if class_name not in crop_counters:
                                crop_counters[class_name] = 0
                            crop_counters[class_name] += 1

                            # Create filename: <page_num>_<class>_<index>.jpg OR <filename>_<class>_<index>.jpg
                            base_name = str(page_num) if page_num else p.stem
                            crop_filename = f"{base_name}_{class_name}_{crop_counters[class_name]}.jpg"
                            crops_dir = save_dir / 'crops'
                            crops_dir.mkdir(parents=True, exist_ok=True)
                            save_one_box(xyxy, imc, file=crops_dir / crop_filename, BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5_300epochs_aug.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/PDF, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=100, help='maximum detections per image')
    parser.add_argument('--device', default='mps', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', default=True, action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default=True, action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', default=[0,1,2,3], nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
