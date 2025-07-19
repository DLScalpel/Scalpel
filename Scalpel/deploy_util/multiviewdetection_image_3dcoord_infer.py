import cv2
import numpy as np
import paddle
from paddle import inference

from deploy_util.predictor import infer_with_onnxruntime_trt_multi


def imnormalize(img, mean, std, to_rgb=True):
    """normalize an image with mean and std.
    """
    # cv2 inplace normalization does not accept uint8
    img = img.copy().astype(np.float32)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def get_resize_crop_shape(img_shape, target_shape):
    H, W = img_shape
    fH, fW = target_shape
    resize = max(fH / H, fW / W)
    resize_shape = (int(W * resize), int(H * resize))
    newW, newH = resize_shape
    crop_h = int(newH) - fH
    crop_w = int(max(0, newW - fW) / 2)

    crop_shape = (crop_h, crop_w, crop_h + fH, crop_w + fW)
    return resize_shape, crop_shape


def get_image(filenames):
    """
    Loads image for a sample
    Args:
        idx [int]: Index of the image sample
    Returns:
        image [np.ndarray(H, W, 3)]: RGB Image
    """
    img = np.stack([cv2.imread(name) for name in filenames], axis=-1)
    imgs = [img[..., i] for i in range(img.shape[-1])]

    new_imgs = []

    target_shape = (320, 800)

    for i in range(len(imgs)):
        img_shape = imgs[i].shape[:2]
        resize_shape, crop_shape = get_resize_crop_shape(
            img_shape, target_shape)

        img = cv2.resize(imgs[i], resize_shape, cv2.INTER_LINEAR)
        img = img[crop_shape[0]:crop_shape[2], crop_shape[1]:crop_shape[3], :]

        new_imgs.append(np.array(img).astype(np.float32))

    mean = np.array([103.530, 116.280, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.120, 58.395], dtype=np.float32)

    new_imgs = [imnormalize(img, mean, std, False) for img in new_imgs]
    return np.array(new_imgs).transpose([0, 3, 1, 2])[np.newaxis, ...]


def run(predictor, img, with_timestamp):
    input_names = predictor.get_input_names()

    input_tensor0 = predictor.get_input_handle(input_names[0])
    input_tensor1 = predictor.get_input_handle(input_names[1])

    num_cams = 6
    if with_timestamp:
        input_tensor2 = predictor.get_input_handle(input_names[2])
        num_cams = 12

    img2lidars = [
        -1.40307297e-03, 9.07780395e-06, 4.84838307e-01, -5.43047376e-02,
        -1.40780103e-04, 1.25770375e-05, 1.04126692e+00, 7.67668605e-01,
        -1.02884378e-05, -1.41007011e-03, 1.02823459e-01, -3.07415128e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
        -9.39000631e-04, -7.65239349e-07, 1.14073277e+00, 4.46270645e-01,
        1.04998052e-03, 1.91798881e-05, 2.06218868e-01, 7.42717385e-01,
        1.48074005e-05, -1.40855671e-03, 7.45946690e-02, -3.16081315e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
        -7.0699735e-04, 4.2389297e-07, -5.5183989e-01, -5.3276348e-01,
        -1.2281288e-03, 2.5626015e-05, 1.0212017e+00, 6.1102939e-01,
        -2.2421273e-05, -1.4170362e-03, 9.3639769e-02, -3.0863306e-01,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
        2.2227580e-03, 2.5312484e-06, -9.7261822e-01, 9.0684637e-02,
        1.9360810e-04, 2.1347081e-05, -1.0779887e+00, -7.9227984e-01,
        4.3742721e-06, -2.2310747e-03, 1.0842450e-01, -2.9406491e-01,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
        5.97175560e-04, -5.88774265e-06, -1.15893924e+00, -4.49921310e-01,
        -1.28312141e-03, 3.58297058e-07, 1.48300052e-01, 1.14334166e-01,
        -2.80917516e-06, -1.41527120e-03, 8.37693438e-02, -2.36765608e-01,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
        3.6048229e-04, 3.8333174e-06, 7.9871160e-01, 4.3321830e-01,
        1.3671946e-03, 6.7484652e-06, -8.4722507e-01, 1.9411178e-01,
        7.5027779e-06, -1.4139183e-03, 8.2083985e-02, -2.4505949e-01,
        0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00
    ]

    if with_timestamp:
        img2lidars += img2lidars

    img2lidars = np.array(img2lidars).reshape([num_cams, 4,
                                               4]).astype('float32')

    input_tensor0.reshape([1, num_cams, 3, 320, 800])
    input_tensor0.copy_from_cpu(img)

    input_tensor1.reshape([num_cams, 4, 4])
    input_tensor1.copy_from_cpu(img2lidars)

    if with_timestamp:
        timestamp = np.zeros([num_cams]).astype('float32')
        timestamp[num_cams // 2:] = 1.0
        input_tensor2.reshape([1, num_cams])
        input_tensor2.copy_from_cpu(timestamp)

    predictor.run()
    outs = []
    output_names = predictor.get_output_names()
    for name in output_names:
        out = predictor.get_output_handle(name)
        out = out.copy_to_cpu()
        out = paddle.to_tensor(out)
        outs.append(out)

    result = {}
    result['pred_boxes'] = outs[0]
    result['pred_scores'] = outs[1]
    result['pred_labels'] = outs[2]

    return result


def multiviewdetection_image_3dcoord_infer(mode,predictor):
    if mode == 'paddlepaddle':
        img_paths = ['/tmp/pycharm_project_403/images/centerpoint.png', '/media/zou/EAGET忆捷/ICSE2026/images/centerpoint.png', '/media/zou/EAGET忆捷/ICSE2026/images/centerpoint.png', '/media/zou/EAGET忆捷/ICSE2026/images/centerpoint.png', '/media/zou/EAGET忆捷/ICSE2026/images/centerpoint.png', '/media/zou/EAGET忆捷/ICSE2026/images/centerpoint.png']
        image = get_image(img_paths)
        num_cams = 6
        img2lidars = [
            -1.40307297e-03, 9.07780395e-06, 4.84838307e-01, -5.43047376e-02,
            -1.40780103e-04, 1.25770375e-05, 1.04126692e+00, 7.67668605e-01,
            -1.02884378e-05, -1.41007011e-03, 1.02823459e-01, -3.07415128e-01,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
            -9.39000631e-04, -7.65239349e-07, 1.14073277e+00, 4.46270645e-01,
            1.04998052e-03, 1.91798881e-05, 2.06218868e-01, 7.42717385e-01,
            1.48074005e-05, -1.40855671e-03, 7.45946690e-02, -3.16081315e-01,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
            -7.0699735e-04, 4.2389297e-07, -5.5183989e-01, -5.3276348e-01,
            -1.2281288e-03, 2.5626015e-05, 1.0212017e+00, 6.1102939e-01,
            -2.2421273e-05, -1.4170362e-03, 9.3639769e-02, -3.0863306e-01,
            0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
            2.2227580e-03, 2.5312484e-06, -9.7261822e-01, 9.0684637e-02,
            1.9360810e-04, 2.1347081e-05, -1.0779887e+00, -7.9227984e-01,
            4.3742721e-06, -2.2310747e-03, 1.0842450e-01, -2.9406491e-01,
            0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
            5.97175560e-04, -5.88774265e-06, -1.15893924e+00, -4.49921310e-01,
            -1.28312141e-03, 3.58297058e-07, 1.48300052e-01, 1.14334166e-01,
            -2.80917516e-06, -1.41527120e-03, 8.37693438e-02, -2.36765608e-01,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
            3.6048229e-04, 3.8333174e-06, 7.9871160e-01, 4.3321830e-01,
            1.3671946e-03, 6.7484652e-06, -8.4722507e-01, 1.9411178e-01,
            7.5027779e-06, -1.4139183e-03, 8.2083985e-02, -2.4505949e-01,
            0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00
        ]

        img2lidars = np.array(img2lidars).reshape([num_cams, 4, 4]).astype('float32')

        image = image.reshape([1, num_cams, 3, 320, 800])
        img2lidars = img2lidars.reshape([num_cams, 4, 4])

        # timestamp = np.zeros([num_cams]).astype('float32')
        # timestamp[num_cams // 2:] = 1.0
        # timestamp = timestamp.reshape([1, num_cams])

        # 把字符串里的.pdmodel去掉
        model = paddle.jit.load(predictor.model_file[:-8])
        result = model(image,img2lidars)

        return result[0]

    elif mode == 'paddleinference':
        paddle_predictor = predictor.get_predictor()
        img_paths = ['/tmp/pycharm_project_403/images/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', '/tmp/pycharm_project_403/images/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', '/tmp/pycharm_project_403/images/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', '/tmp/pycharm_project_403/images/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', '/tmp/pycharm_project_403/images/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', '/tmp/pycharm_project_403/images/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg']
        image = get_image(img_paths)
        num_cams = 6
        img2lidars = [
            -1.40307297e-03, 9.07780395e-06, 4.84838307e-01, -5.43047376e-02,
            -1.40780103e-04, 1.25770375e-05, 1.04126692e+00, 7.67668605e-01,
            -1.02884378e-05, -1.41007011e-03, 1.02823459e-01, -3.07415128e-01,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
            -9.39000631e-04, -7.65239349e-07, 1.14073277e+00, 4.46270645e-01,
            1.04998052e-03, 1.91798881e-05, 2.06218868e-01, 7.42717385e-01,
            1.48074005e-05, -1.40855671e-03, 7.45946690e-02, -3.16081315e-01,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
            -7.0699735e-04, 4.2389297e-07, -5.5183989e-01, -5.3276348e-01,
            -1.2281288e-03, 2.5626015e-05, 1.0212017e+00, 6.1102939e-01,
            -2.2421273e-05, -1.4170362e-03, 9.3639769e-02, -3.0863306e-01,
            0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
            2.2227580e-03, 2.5312484e-06, -9.7261822e-01, 9.0684637e-02,
            1.9360810e-04, 2.1347081e-05, -1.0779887e+00, -7.9227984e-01,
            4.3742721e-06, -2.2310747e-03, 1.0842450e-01, -2.9406491e-01,
            0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
            5.97175560e-04, -5.88774265e-06, -1.15893924e+00, -4.49921310e-01,
            -1.28312141e-03, 3.58297058e-07, 1.48300052e-01, 1.14334166e-01,
            -2.80917516e-06, -1.41527120e-03, 8.37693438e-02, -2.36765608e-01,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
            3.6048229e-04, 3.8333174e-06, 7.9871160e-01, 4.3321830e-01,
            1.3671946e-03, 6.7484652e-06, -8.4722507e-01, 1.9411178e-01,
            7.5027779e-06, -1.4139183e-03, 8.2083985e-02, -2.4505949e-01,
            0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00
        ]

        img2lidars = np.array(img2lidars).reshape([num_cams, 4, 4]).astype('float32')

        image = image.reshape([1, num_cams, 3, 320, 800])
        img2lidars = img2lidars.reshape([num_cams, 4, 4])
        # 含有历史信息的模型with_timestamp为True
        result = run(paddle_predictor, image,img2lidars)
        return result['pred_boxes']
    elif mode == 'autoware':
        paddle_predictor = predictor.get_predictor()
        img_paths = ['/tmp/pycharm_project_403/images/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', '/tmp/pycharm_project_403/images/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', '/tmp/pycharm_project_403/images/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', '/tmp/pycharm_project_403/images/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', '/tmp/pycharm_project_403/images/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', '/tmp/pycharm_project_403/images/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg']
        image = get_image(img_paths)
        num_cams = 6
        img2lidars = [
            -1.40307297e-03, 9.07780395e-06, 4.84838307e-01, -5.43047376e-02,
            -1.40780103e-04, 1.25770375e-05, 1.04126692e+00, 7.67668605e-01,
            -1.02884378e-05, -1.41007011e-03, 1.02823459e-01, -3.07415128e-01,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
            -9.39000631e-04, -7.65239349e-07, 1.14073277e+00, 4.46270645e-01,
            1.04998052e-03, 1.91798881e-05, 2.06218868e-01, 7.42717385e-01,
            1.48074005e-05, -1.40855671e-03, 7.45946690e-02, -3.16081315e-01,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
            -7.0699735e-04, 4.2389297e-07, -5.5183989e-01, -5.3276348e-01,
            -1.2281288e-03, 2.5626015e-05, 1.0212017e+00, 6.1102939e-01,
            -2.2421273e-05, -1.4170362e-03, 9.3639769e-02, -3.0863306e-01,
            0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
            2.2227580e-03, 2.5312484e-06, -9.7261822e-01, 9.0684637e-02,
            1.9360810e-04, 2.1347081e-05, -1.0779887e+00, -7.9227984e-01,
            4.3742721e-06, -2.2310747e-03, 1.0842450e-01, -2.9406491e-01,
            0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00,
            5.97175560e-04, -5.88774265e-06, -1.15893924e+00, -4.49921310e-01,
            -1.28312141e-03, 3.58297058e-07, 1.48300052e-01, 1.14334166e-01,
            -2.80917516e-06, -1.41527120e-03, 8.37693438e-02, -2.36765608e-01,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
            3.6048229e-04, 3.8333174e-06, 7.9871160e-01, 4.3321830e-01,
            1.3671946e-03, 6.7484652e-06, -8.4722507e-01, 1.9411178e-01,
            7.5027779e-06, -1.4139183e-03, 8.2083985e-02, -2.4505949e-01,
            0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00
        ]

        img2lidars = np.array(img2lidars).reshape([num_cams, 4, 4]).astype('float32')

        image = image.reshape([1, num_cams, 3, 320, 800])
        img2lidars = img2lidars.reshape([num_cams, 4, 4])
        # 含有历史信息的模型with_timestamp为True
        result = infer_with_onnxruntime_trt_multi('/tmp/pycharm_project_403/exported_model/smoke.onnx',image,img2lidars)
        return result

