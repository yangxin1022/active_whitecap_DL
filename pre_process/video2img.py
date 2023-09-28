import cv2
import os

def video2img(video_path,img_path,
              img_name='',
              start_sec=0,frame_rate=60): 
    """This function is used to convert MP4 video
    into images at a given FPS. FPS is 60 Hz by default.

    Args:
        video_path (str): The path of the video
        img_path (str): The path where the images are saved
        img_name (str, optional): Define the prefix of the image names
        start_sec (int, optional): start time. Defaults to 0.
        frame_rate (int, optional): FPS of ouput images. Defaults to 60.
    """
  
    if not os.path.exists(video_path):
        print('The video path does not exist')
    else:
        video_cap = cv2.VideoCapture(video_path)
        open_video = video_cap.isOpened()
        print(f'The video can be opened: {open_video}')
        
    if open_video:
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f'fps={int(fps)}, Frames={int(frames)}')
        
        count = 0
        flag = True
        while flag:
            # set the current position of the video (milliseconds)
            video_cap.set(cv2.CAP_PROP_POS_MSEC, start_sec*1000)
            flag, image = video_cap.read()
            if not flag:
                print('Processed finished!')
                break
            cv2.imwrite(img_path+'/'+img_name+
                        f'{count}.png', image)
            print('Output a new frame: ',flag)
            count += 1
            start_sec = start_sec + 1/frame_rate
        video_cap.release()
        
if __name__ == '__main__':
    video_path = '/Users/xinyang/Library/CloudStorage/OneDrive-TexasA&MUniversity/Code_sorted_PIV/Result/1/2_6_GH090009_0011.MP4'
    img_path = '/Users/xinyang/Desktop/test'
    img_name = 'img_'
    video2img(video_path,img_path,img_name,start_sec=0,frame_rate=60)
    print('The video has been transformed into images. Video path is f{video_path}. Images are saved in f{img_path}')
        