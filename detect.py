import argparse
import os
import sys
import time
from pathlib import Path
import PIL.ImageOps
from PIL import Image
from PIL import ImageEnhance
import cv2
import numpy as np
import tools.infer.predict_rec5_final as ocr
import torch
import torch.backends.cudnn as cudnn
import models.crnn as crnn


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
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
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories 检测结果保存目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize 初始化 运行设备
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model  加载模型
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix 检查权重的后缀是否可接受
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans 后端布尔运算
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults 指定默认值
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        # names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        names = ['红', '绿', '黄', '前-红', '前-绿', '前-黄', '左-红', '左-绿', '左-黄',
                 '右-红', '右-绿', '右-黄', '___']
        names1 = '数字'
        if half:
            model.half()  # to FP16


        if classify:  # second-stage classifier
            modelc = crnn.CRNN(32, 1, 37, 256)  # initialize
            modelc.load_state_dict(torch.load('english_crnn.pth', map_location=device)['model']).to(device).eval()

    imgsz = check_img_size(imgsz, s=stride)  # check image size

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

    # Run inference 运行推理
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0

    jishuqi = 0
    for path, img, im0s, vid_cap in dataset:
        #path 路径  img：480*640*3 经过640变换的图片 im0s:502*678*3 原始图片
        # print('path,img,im0s,vid_cap',path,img.shape,im0s.shape,vid_cap)
        imgg = img.copy()
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()  # 时间计算
        dt[0] += t2 - t1  # 把时间保存在dt中

        # Inference 推理（执行预测 得到pred文件）
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
            ################################
            # print('pred.shape-----------',type(pred),pred.shape)
            ################################
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS 非极大抑制
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional) #第二步分类！
        # classify = True
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions 预测程序
        for i, det in enumerate(pred):  # per image # pred里为第几个框 和 相应的labels（框和置信度、分类结果） i不知道为啥一直是0
            # print('i,det',i,det)
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            #############################################################################################################
            # 得到的det为tensor格式的矩阵，n个框则n行，6列（4+1+1 位置 置信度 分类）！！！！！！！！！！！！！！！！！！！！！！
            #############################################################################################################

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # print(type(det))
                # Print results 打印结果
                for c in det[:, -1].unique():  # 此处c为det矩阵的最后一列值，即分类结果
                    # print('c的结果是',c) # # c的结果是 tensor(1., device='cuda:0') 1 为类别
                    n = (det[:, -1] == c).sum()  # detections per class
                    # print('n的结果是',n) # n的结果是 tensor(1, device='cuda:0')

                    if c != 12:
                        s += f"{n} 个 {names[int(c)]} {'s' * (n > 1)}, "  # add to string
                    else:

                        '''在此处修改’数字‘,即names1，设置一个变量用来存具体值，当c=12时，使用新的神经网络预测，并修改该变量'''

                        # names1 = predict_data(cut_img)
                        #####################################################################
                        s += f"{n} 个 {names1} {'s' * (n > 1)}, "  # add to string 待修改
                        #####################################################################
                    # print('s的结果是', s) # s的结果是 480x640 1 绿

                # Write results 画出图象

                for *xyxy, conf, cls in reversed(det):
                    '''
                    这里xyxy，conf，cls分别为位置 置信度 类别
                    '''
                    if save_txt:  # Write to file 保存txt文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image 在图上绘制bbox
                        c = int(cls)  # integer class
                        if c == 12:# 当结果为数码管时
                            # print('xyxy 是',xyxy) #左上角xy和右下角xy的list，包含4个tensor
                            # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            # x, y, w, h = xywh
                            x1,y1,x2,y2 = xyxy
                            # print('x1,y1,x2,y2=',x1,y1,x2,y2)
                            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                            crop = im0s[y1:y2,x1:x2,:] # 高，宽，channel

                            # img = Image.fromarray(crop)
                            # crop.show()
                            # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) # 转灰度图
                            # th, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #ostu
                            # # crop = crop.reshape(crop.shape[0],crop.shape[1],1)
                            # crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

                            # cv2.imshow('crop',crop)
                            # cv2.waitKey(0)
                            # cv2.imwrite('data/cut/'+str(jishuqi)+'.jpg',crop)
                            # jishuqi += 1
                            # print('jishuqi====',jishuqi)


#--------------------------------------------------------------------------------------------------------------------
                          # PLAN A : 使用CRNN
                          #   print(crop.shape) # （个数，通道，高，宽）
                          #   cropp = crop.to('cpu') # 这样crop在gpu cropp在cpu
                            # print(cropp)
                            # cropp = cropp.numpy() # 得到np格式的cropp截图


                            names1 = ocr.ocr(crop)
                            names1 = names1[0][0]
                            print('names1:',names1)
                            # names1 = '新的值' # names1 = model(x) CRNN的地方
# -------------------------------------------------------------------------------------------------------------------
                            label = None if hide_labels else (names1 if hide_conf else f'{names1} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(12, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        else:
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')

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
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

# x-model 3:08  ; l-model 2:08  ; m-mode:1:36 ； s-model 1：15
'''
注意选择的模型和文件路径
'''
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'model_s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'input_data', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'output_data', help='save results to project/name')
    parser.add_argument('--name', default='result', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
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
