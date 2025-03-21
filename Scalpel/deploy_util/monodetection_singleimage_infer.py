import cv2
import numpy as np
import paddle


def get_ratio(ori_img_size, output_size, down_ratio=(4, 4)):
    return np.array([[
        down_ratio[1] * ori_img_size[1] / output_size[1],
        down_ratio[0] * ori_img_size[0] / output_size[0]
    ]], np.float32)


def get_img(img_path):
    img = cv2.imread(img_path)
    origin_shape = img.shape
    img = cv2.resize(img, (1280, 384))

    target_shape = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0
    img = np.subtract(img, np.array([0.485, 0.456, 0.406]))
    img = np.true_divide(img, np.array([0.229, 0.224, 0.225]))
    img = np.array(img, np.float32)

    img = img.transpose(2, 0, 1)
    img = img[None, :, :, :]

    return img, origin_shape, target_shape


def run(predictor, image, K, down_ratio):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        if name == "images":
            input_tensor.reshape(image.shape)
            input_tensor.copy_from_cpu(image.copy())
        elif name == "trans_cam_to_img":
            input_tensor.reshape(K.shape)
            input_tensor.copy_from_cpu(K.copy())
        elif name == "down_ratios":
            input_tensor.reshape(down_ratio.shape)
            input_tensor.copy_from_cpu(down_ratio.copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results
def monodetection_singleimage_infer(mode,predictor):
    image_path = '/media/zou/EAGET忆捷/ICSE2026/images/centerpoint.png'
    if mode == 'paddlepaddle':
        # Listed below are camera intrinsic parameter of the kitti dataset
        # If the model is trained on other datasets, please replace the relevant data
        K = np.array([[[721.53771973, 0., 609.55932617],
                       [0., 721.53771973, 172.85400391], [0, 0, 1]]], np.float32)

        img, ori_img_size, output_size = get_img(image_path)
        ratio = get_ratio(ori_img_size, output_size)
        # 把字符串里的.pdmodel去掉
        model = paddle.jit.load(predictor.model_file[:-8])
        results = model(ratio,img,K)
        return results
    elif mode == 'paddleinference':
        # Listed below are camera intrinsic parameter of the kitti dataset
        # If the model is trained on other datasets, please replace the relevant data
        K = np.array([[[721.53771973, 0., 609.55932617],
                       [0., 721.53771973, 172.85400391], [0, 0, 1]]], np.float32)

        img, ori_img_size, output_size = get_img(image_path)
        ratio = get_ratio(ori_img_size, output_size)

        results = run(predictor.get_predictor(), img, K, ratio)

        return results[0]