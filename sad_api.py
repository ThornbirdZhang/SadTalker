import array
import os
from app_sadtalker import ref_video_fn

#from pytorch_lightning import seed_everything

#for fastapi
from fastapi import FastAPI , Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import asyncio
import threading
import logging
import urllib.request
import requests
import datetime
from moviepy.editor import VideoFileClip

from src.gradio_demo import SadTalker

logging.basicConfig(
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    format='[%(asctime)s %(levelname)-7s (%(name)s) <%(process)d> %(filename)s:%(lineno)d] %(message)s',
    level=logging.INFO
)


class MyClass:
    pass



class SadRequest(BaseModel):
    pose_style: int = 0  # 0- 45
    size_of_image: int = 512 # 256,512
    preprocess_type: str = 'full' #'crop', 'resize','full', 'extcrop', 'extfull'
    is_still_mode: bool = True
    enhancer: bool = True #quite slow,
    #use_idle_mode: bool #not sure its meaning
    #length_of_audio: int  #with use_idle_mode
    use_ref_video: bool = False #activate ref_video settings
    ref_video: str  = "" #local file path,
    ref_info: str = 'all' #pose, blink, pose+blink, all
    use_blink: bool = True #default
    exp_scale: float = 1.0 #expression scale
    image_url: str = ""
    audio_url: str = ""

class SadActor:
    def __init__(self, name: str):
        self.name = name
        #better in config, need modification for every node
        self.tmp_folder = "./tmp"
        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)
            logging.info(f"created tmp folder {self.tmp_folder}")
        else:
            logging.info(f"tmp folder {self.tmp_folder} exists")

        self.www_folder = "/data/SadTalker/results"
        public_ip = self.get_public_ip()
        logging.info(f"public ip for this module is {public_ip}")
        self.url_prefix = "http://" + public_ip + ":8076/"

        self.version = "sad_v2"

        self.sad_talker = SadTalker('checkpoints', 'src/config', lazy_load=True)
        self.task_id = None
        self.result = 0 # 0, unknown; -1, failed; 1: success
        self.status = 0 #0, init/empty; 1, doing
        self.msg = "" #error msg
        self.result_code = 100 # based on xme. 
        self.result_url = ""
        self.result_file = ""
        self.result_length = 1
        self.sad_request = SadRequest()

    def say_hello(self):
        logging.debug(f"Hello, {self.name}!")
    
    def get_public_ip(self):
        response = requests.get('https://ifconfig.me/ip')
        return response.text

    def init_task(self, content: SadRequest):
        self.status = 1  #locked
        self.task_id = self.name + datetime.datetime.now().strftime("%f")
        self.result = 0 # 0, unknown; -1, failed; 1: success
        self.msg = "" #error msg
        self.result_code = 100 # based on xme. 
        self.result_url = ""
        self.sad_request.audio_url = content.audio_url
        self.sad_request.enhancer = content.enhancer
        self.sad_request.image_url = content.image_url
        self.sad_request.is_still_mode = content.is_still_mode
        self.sad_request.pose_style = content.pose_style
        self.sad_request.preprocess_type = content.preprocess_type
        self.sad_request.ref_info = content.ref_info
        self.sad_request.size_of_image = content.size_of_image
        self.sad_request.use_blink = content.use_blink
        self.sad_request.use_ref_video = content.use_ref_video
        self.sad_request.ref_video = content.ref_video
        self.sad_request.exp_scale = content.exp_scale
        logging.info("after init_task")

    def start_task(self):
        logging.info("before start_task")
        self.do_sample()
        return

    #download url to folder, keep the file name untouched
    def download(self, url: str, directory:str):
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = url.split("/")[-1]

        file_name = os.path.join(directory, filename)
        urllib.request.urlretrieve(url, file_name)
        return file_name

    #action function, url is the http photo
    def do_sample(self):
        #empty? checked before, no need
         try:
             logging.info(f"download photo file:{self.sad_request.image_url} to {self.tmp_folder}, {self.sad_request.audio_url} to {self.tmp_folder} ")
             photo_file = self.download(self.sad_request.image_url, self.tmp_folder)
             audio_file = self.download(self.sad_request.audio_url, self.tmp_folder)
             logging.info(f"downloaded photo and audio ")
             ref_video_file = ""
             if(self.sad_request.use_ref_video and self.sad_request.ref_video != "" ) :
                 ref_video_file = self.download(self.sad_request.ref_video, self.tmp_folder)
                 logging.info(f"downloaded ref video to {ref_video_file}")

             #start inference
             output_video = self.sad_talker.test(photo_file, audio_file, self.sad_request.preprocess_type, self.sad_request.is_still_mode, self.sad_request.enhancer,2, \
                                                 self.sad_request.size_of_image,self.sad_request.pose_style  \
                                                 ,self.sad_request.exp_scale, self.sad_request.use_ref_video, ref_video_file, self.sad_request.ref_info, \
                                                 use_blink=self.sad_request.use_blink, result_dir=self.www_folder)
             logging.info(f"finished inferencing, output={output_video}")
             new_output_video = output_video.replace("#", "")
             os.rename(output_video, new_output_video)
             output_video = new_output_video

             self.result_file = output_video
             video = VideoFileClip(self.result_file)
             self.result_length = video.duration
             logging.info(f"result video={self.result_file}, length = {self.result_length}")

             #for output url 
             diff = os.path.relpath(self.result_file, self.www_folder)
             self.result_url = self.url_prefix + diff
             logging.info(f'save_path={output_video}, www_folder={self.www_folder}, result_url={self.result_url}, diff={diff}')
             self.result = 1
             self.status = 0
             self.result_code = 100
             self.msg = "succeeded"

         except Exception as e:
             logging.error(f"something wrong during task={self.task_id}, exception={repr(e)}")
             self.result_url = ""
             self.result = -1
             self.status = 0
             self.result_code = 103
             self.msg = "something wrong during task=" + self.task_id + ", please contact admin."
         finally:
             self.status = 0

    def get_status(self, task_id: str):
        ret = MyClass()
        length = 1
        if(task_id != self.task_id):
            #not the current task
            ret.result_url = ""
            ret.result_code = 200
            ret.msg = "cannot find task_id=" + task_id
        else:
            ret.result_url = self.result_url;
            if(self.result == 0):
                ret.result_code = 102
                ret.msg = "task(" + task_id + ") is running."
            elif(self.result == 1): 
                ret.result_code = 100
                ret.msg = "task(" + task_id + ") has succeeded."

            elif(self.result == -1): 
                ret.result_code = 103
                ret.msg = "task(" + task_id + ") has failed."
            else:
                ret.result_code = 103
                ret.msg = "task(" + task_id + ") has failed for uncertainly."     
        
        retJ = {"result_url": ret.result_url, "result_code": ret.result_code, "msg": ret.msg,"api_time_consume":self.result_length, "api_time_left":0, "video_w":0, "video_h":0, "gpu_type":"", "gpu_time_estimate":0, "gpu_time_use":0}
        #retJson = json.dumps(retJ)
        logging.debug(f"get_status for task_id={task_id}, return {retJ}" )
        return retJ



app = FastAPI()
sadActor = SadActor("sad_node_100")


@app.get("/")
async def root():
    return {"message": "Hello World, sad, May God Bless You."}

@app.post("/api/phototalking/startTask")
async def post_t2tt(content : SadRequest):
    logging.info(f"before infer, content= {content}")
    result = MyClass()

    if(sadActor.status != 0):
        logging.warn(f"engine is busy with task={sadActor.task_id}, cannot accept more.")
        result.task_id = ""
        result.result_code = 203
        result.msg = "engine is busy with task, cannot accept more."
    else:
        sadActor.init_task(content)
        result.task_id = sadActor.task_id
        result.result_code = 100
        result.msg = "task_id=" + sadActor.task_id + " has started."
        loop = asyncio.get_event_loop()
        thread = threading.Thread(target = sadActor.start_task)
        thread.daemon = True
        thread.start()
        

    retJ = {"task_id":result.task_id, "result_code": result.result_code, "msg": result.msg}
    #response = Response(content=retJ, media_type="application/json")
    #retJson = json.dumps(retJ)
    logging.info(f"url={content.image_url}, task_id={result.task_id}, return {retJ}")

    #return response
    return retJ

@app.get("/api/phototalking/startTask")
async def get_status(taskID:str):
    logging.info(f"before startTask, taskID= {taskID}")
    return sadActor.get_status(taskID)
