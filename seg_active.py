import numpy as np
import os
import cv2
import warnings
import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as functional

# Load local files
import model.u_net as u_net
import util.toolbox
from pre_process.rectification import rectify
from pre_process.rectification import get_ROI






if __name__ == '__main__':
    ##############################################
    # must check all the camera related parameters
    # intrinsic matrix of the camera
    IntrinsicMatrix = np.array([[1.3533e3, 0, 1.1605e3], 
                                [0, 1.418e3, 529.8685], 
                                [0, 0, 1]])

    # camera height (m)
    H = 7.0
    # 4.5 m for the lower one. 

    # angle of camera when it is stable (rad)
    incStab = 73 * np.pi / 180
    rollStab = 0 * np.pi / 180
    aziStab = 0
    ############################################### 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the hyperparameters from json file
    model_dir = 'E:/my-whitecaps/code/organized/model'
    # "Directory containing params.json"
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(
        json_path
    ), f"No json configuration file found at {json_path}"
    params = util.toolbox.Params(json_path)
    model = u_net.UNet(params).float()
    # Load the model parameters
    model.load_state_dict(
        torch.load("E:/my-whitecaps/code/model_cifar_200.pt"))
    model.to(device)

    store_path = "E:\GoPro_1Hz"
    date_folder = [
        file
        for file in os.listdir(store_path)
        if os.path.isdir(os.path.join(store_path, file))
    ]
    for i in range(1):
        i = i + 2
        # get the name of the folders that store one of these cameras
        camera_folder = os.listdir(os.path.join(store_path, date_folder[i]))


        for camera_i in camera_folder:
            video_path = os.path.join(store_path, date_folder[i], camera_i)
            video_folder = os.listdir(video_path)
            for video_id in video_folder:
                image_path = os.path.join(video_path, video_id)
                image_file = os.listdir(image_path)
                result = pd.DataFrame(columns=('date','camera','video','image_fname',
                                    'active_whitecaps','active_whitecaps_fraction',
                                    'predict_image'))

                for fname in image_file[:]:
                    if fname[:2] == '._':
                        os.remove(os.path.join(image_path, fname))
                        image_file.remove(fname)

                image_file.sort(key=lambda x:int(x[14:-4]))


                print(f'The code is working on {date_folder[i]}-{camera_i}-{video_id}!!!')
                for fname in image_file:

                    path = os.path.join(image_path, fname)
                    img = cv2.imread(path)

                    try:
                        roi_up, roi_bottom, line, rho, theta = get_ROI(img)
                    except:
                        result = result.append(pd.DataFrame({
                        'date':[date_folder[i]],
                        'camera':[camera_i],
                        'video':[video_id],
                        'image_fname':[fname],
                        'active_whitecaps':[0],
                        'active_whitecaps_fraction':[0],
                        'predict_image':[0]
                        }))
                    else:

                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        img_roi = img[roi_up: roi_bottom+1, 0:1920]
                        img_roi_tensor = torch.from_numpy(img_roi / 255).float()
                        img_roi_tensor = img_roi_tensor.unsqueeze(0)
                        img_roi_tensor = img_roi_tensor.unsqueeze(0)
                        model.to(device)
                        img_roi_tensor = img_roi_tensor.to(device)

                        with torch.no_grad():
                            output = model(img_roi_tensor)
                        output = output.squeeze(0)
                        output = output.squeeze(0)

                        # assign all large than 0.8 as 1 and less as 0
                        output = output.cpu().numpy()
                        output[output >= 0.8] = 1
                        output[output < 0.8] = 0

                        output_BGR = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                        output_rectified = rectify(img,IntrinsicMatrix,H,incStab,rollStab,
                                                aziStab,line,rho,theta,output)

                        img_roi_rec = rectify(img,IntrinsicMatrix,H,incStab,rollStab,
                                            aziStab,line,rho,theta,img_roi)
                        area = img_roi_rec[img_roi_rec>0].size
                        active_whitecaps = np.argwhere(output_rectified>0)
                        active_whitecaps_area = output_rectified[output_rectified>0].size
                        active_whitecaps_fraction = active_whitecaps_area / area
                        result = result.append(pd.DataFrame({
                            'date':[date_folder[i]],
                            'camera':[camera_i],
                            'video':[video_id],
                            'image_fname':[fname],
                            'active_whitecaps':[active_whitecaps],
                            'active_whitecaps_fraction':[active_whitecaps_fraction],
                            'predict_image':[output]
                        }))
                result = result.reset_index(drop=True)
                if not os.path.exists(f"E:/GoPro_1Hz_Unet_output/{date_folder[i]}/{camera_i}"):
                    os.makedirs(f"E:/GoPro_1Hz_Unet_output/{date_folder[i]}/{camera_i}")
                result.to_csv(
                    f"E:/GoPro_1Hz_Unet_output/{date_folder[i]}/{camera_i}/{video_id[:-4]}_active.csv"
                )
        