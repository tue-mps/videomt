import os
from PIL import Image
from multiprocessing import Pool


DIR = '../VIPSeg/imgs'
Target_Dir = '../VIPSeg/VIPSeg_720P'


def change(DIR,video,image):
    if os.path.isfile(os.path.join(Target_Dir,'images',video,image)) and os.path.isfile(os.path.join(Target_Dir,'panomasks',video,image.split('.')[0]+'.png')):
        return

    img = Image.open(os.path.join(DIR,video,image))
    w,h = img.size
    img = img.resize((int(720*w/h),720),Image.BILINEAR)

    if not os.path.exists(os.path.join(Target_Dir,'images',video)):
        os.makedirs(os.path.join(Target_Dir,'images',video))

    img.save(os.path.join(Target_Dir,'images',video,image))
    print('Processing video {} image {}'.format(video,image))

p = Pool(28)
ori_videos = os.listdir(DIR)
avaliable_videos = os.listdir(os.path.join(Target_Dir, 'images'))
need_convert_videos = []
for video in ori_videos:
    if video not in avaliable_videos:
        need_convert_videos.append(video)
need_convert_videos.sort()
for video in need_convert_videos:
    if video[0]=='.':
        continue
    for image in sorted(os.listdir(os.path.join(DIR,video))):
        if image[0]=='.':
            continue
        change(DIR,video,image)
print('finish')

